// SPDX-License-Identifier: GPL-2.0
/*
 * ROCm XIO I/O engine — GPU-initiated NVMe using AMD rocm-xio (nvme-ep).
 *
 * Each fio I/O is submitted as a one-operation HIP kernel (high per-op
 * overhead vs xio-tester batching). Requires the rocm-xio out-of-tree
 * kernel module, ROCm, and librocm-xio (see HOWTO.rst).
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

extern "C" {
#include "../fio.h"
#include "../log.h"
#include "../optgroup.h"
}

#include <linux/nvme_ioctl.h>
#include <sys/ioctl.h>

#include <xio.h>
#include <nvme-ep.h>

#include "nvme.h"

namespace {

constexpr unsigned kMinQueueDepth = 2;

struct rocm_xio_options {
	void *pad;
	int gpu_id;
	unsigned int queue_id;
	unsigned int queue_depth;
	unsigned int nsid;
	unsigned int memory_mode;
	unsigned int use_device_mem;
	char *kmod_path;
	char *controller_override;
};

struct fio_rocm_xio_thread {
	uint64_t *prp_pool;
	uint64_t prp_pool_dma;
	size_t prp_pool_bytes;
};

struct fio_rocm_xio_file {
	uint32_t lba_bytes;
	uint32_t nsid;
	uint64_t nlba;
	char *controller_path;
	struct xioBufferInfo data_buf;
	bool data_buf_valid;
	size_t data_buf_size;
	struct nvme_queue_info qinfo;
	bool qinfo_valid;
	void *bar0_cpu;
	void *bar0_gpu;
	uint16_t nvme_bdf;
};

static struct rocm_xio_options *eo(struct thread_data *td)
{
	return static_cast<struct rocm_xio_options *>(td->eo);
}

static struct fio_rocm_xio_thread *thread_priv(struct thread_data *td)
{
	return static_cast<struct fio_rocm_xio_thread *>(td->io_ops_data);
}

static uint32_t next_pow2_u32(uint32_t v)
{
	if (v <= 1)
		return 1;
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v + 1;
}

static int parse_ns_path(const char *path, char *controller, size_t ctl_len,
			 unsigned *nsid_out)
{
	unsigned ctl, ns;

	if (!path)
		return -EINVAL;
	if (sscanf(path, "/dev/nvme%un%u", &ctl, &ns) != 2) {
		log_err("rocm-xio: filename must be /dev/nvme%%un%%u (namespace), "
			"got %s\n",
			path);
		return -EINVAL;
	}
	if (snprintf(controller, ctl_len, "/dev/nvme%u", ctl) >= (int)ctl_len)
		return -ENAMETOOLONG;
	*nsid_out = ns;
	return 0;
}

static int nvme_identify_ns(int fd, uint32_t nsid, struct nvme_id_ns *ns)
{
	struct nvme_passthru_cmd cmd = {
		.opcode = nvme_admin_identify,
		.nsid = nsid,
		.addr = (__u64)(uintptr_t)ns,
		.data_len = NVME_IDENTIFY_DATA_SIZE,
		.cdw10 = NVME_IDENTIFY_CNS_NS,
		.cdw11 = NVME_CSI_NVM << NVME_IDENTIFY_CSI_SHIFT,
		.timeout_ms = NVME_DEFAULT_IOCTL_TIMEOUT,
	};

	return ioctl(fd, NVME_IOCTL_ADMIN_CMD, &cmd);
}

static int fio_rocm_xio_setup(struct thread_data *td)
{
	struct rocm_xio_options *o = eo(td);

	if (o->queue_depth == 0)
		o->queue_depth = 64;
	if (o->queue_depth < kMinQueueDepth ||
	    (o->queue_depth & (o->queue_depth - 1)) != 0) {
		log_err("rocm-xio: queue_depth must be a power of 2 and >= %u\n",
			kMinQueueDepth);
		return 1;
	}
	return 0;
}

static int fio_rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o = eo(td);
	hipError_t e = hipSetDevice(o->gpu_id);

	if (e != hipSuccess) {
		log_err("rocm-xio: hipSetDevice(%d): %s\n", o->gpu_id,
			hipGetErrorString(e));
		return 1;
	}
	return 0;
}

static int fio_rocm_xio_post_init(struct thread_data *td)
{
	struct fio_rocm_xio_thread *tp;
	uint64_t max_bs;
	uint32_t xfer_pages;
	uint32_t prp_entries;
	uint32_t prp_stride;
	size_t pool_bytes;
	long pgsz;
	size_t align;
	int pr;

	max_bs = td->o.max_bs[DDIR_READ];
	if (td->o.max_bs[DDIR_WRITE] > max_bs)
		max_bs = td->o.max_bs[DDIR_WRITE];
	if (max_bs == 0)
		max_bs = 4096;

	xfer_pages = (uint32_t)((max_bs + NVME_PAGE_SIZE - 1) / NVME_PAGE_SIZE);
	if (xfer_pages > NVME_PAGE_SIZE / (uint32_t)sizeof(uint64_t) + 1) {
		log_err("rocm-xio: block size too large for PRP list\n");
		return 1;
	}

	prp_entries = (max_bs > NVME_PAGE_SIZE)
			  ? (uint32_t)((max_bs / NVME_PAGE_SIZE) + 1)
			  : 0;
	prp_stride = prp_entries;
	if (prp_stride > 1)
		prp_stride = next_pow2_u32(prp_stride);

	pgsz = sysconf(_SC_PAGESIZE);
	align = (pgsz > 0) ? (size_t)pgsz : 4096U;
	pool_bytes = (size_t)prp_stride * sizeof(uint64_t);
	pool_bytes = (pool_bytes + align - 1) & ~(align - 1);

	tp = static_cast<struct fio_rocm_xio_thread *>(calloc(1, sizeof(*tp)));
	if (!tp)
		return 1;

	if (prp_stride > 0) {
		pr = posix_memalign(reinterpret_cast<void **>(&tp->prp_pool),
				    align, pool_bytes);
		if (pr != 0 || !tp->prp_pool) {
			free(tp);
			return 1;
		}
		memset(tp->prp_pool, 0, pool_bytes);
		tp->prp_pool_bytes = pool_bytes;
	}

	td->io_ops_data = tp;
	return 0;
}

static int fio_rocm_xio_iomem_alloc(struct thread_data *td, size_t total_mem)
{
	struct rocm_xio_options *o = eo(td);

	if (o->use_device_mem) {
		td->o.mem_type = MEM_CUDA_MALLOC;
		td->o.mem_align = 4096;
	} else {
		td->o.mem_type = MEM_MALLOC;
		td->o.mem_align = (unsigned int)getpagesize();
	}
	td->o.mem_size = total_mem;
	return 0;
}

static int fio_rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_options *o = eo(td);
	struct fio_rocm_xio_file *xf;
	char controller[PATH_MAX];
	unsigned path_nsid = 0;
	int ret;
	struct nvme_id_ns ns;
	int ctl_fd = -1;
	void *bar_cpu = nullptr;
	void *bar_gpu = nullptr;

	if (f->filetype != FIO_TYPE_BLOCK && f->filetype != FIO_TYPE_CHAR) {
		log_err("rocm-xio: need block or char NVMe namespace device\n");
		return 1;
	}

	xf = static_cast<struct fio_rocm_xio_file *>(calloc(1, sizeof(*xf)));
	if (!xf)
		return 1;

	if (o->controller_override && o->controller_override[0]) {
		if (strlen(o->controller_override) >= sizeof(controller)) {
			free(xf);
			return 1;
		}
		strcpy(controller, o->controller_override);
	} else {
		ret = parse_ns_path(f->file_name, controller, sizeof(controller),
				    &path_nsid);
		if (ret < 0) {
			free(xf);
			return 1;
		}
	}

	ctl_fd = open(controller, O_RDWR);
	if (ctl_fd < 0) {
		log_err("rocm-xio: open %s: %s\n", controller, strerror(errno));
		free(xf);
		return 1;
	}

	if (o->nsid)
		xf->nsid = o->nsid;
	else if (path_nsid)
		xf->nsid = path_nsid;
	else {
		int ns = ioctl(ctl_fd, NVME_IOCTL_ID);

		if (ns < 0) {
			log_err("rocm-xio: NVME_IOCTL_ID failed on %s\n",
				controller);
			close(ctl_fd);
			free(xf);
			return 1;
		}
		xf->nsid = (unsigned)ns;
	}

	memset(&ns, 0, sizeof(ns));
	ret = nvme_identify_ns(ctl_fd, xf->nsid, &ns);
	close(ctl_fd);
	if (ret) {
		log_err("rocm-xio: identify namespace failed (%d)\n", ret);
		free(xf);
		return 1;
	}

	xf->lba_bytes = 1U << ns.lbaf[(ns.flbas & 0xf)].ds;
	xf->nlba = ns.nsze;
	xf->controller_path = strdup(controller);
	if (!xf->controller_path) {
		free(xf);
		return 1;
	}

	ret = xio::detectBdfFromDevice(controller, &xf->nvme_bdf);
	if (ret) {
		log_err("rocm-xio: detectBdfFromDevice: %s\n", strerror(-ret));
		free(xf->controller_path);
		free(xf);
		return 1;
	}

	ret = xio::mapPciBar(xf->nvme_bdf, 0, &bar_cpu, &bar_gpu, 8192);
	if (ret < 0) {
		log_err("rocm-xio: mapPciBar: %s\n", strerror(-ret));
		free(xf->controller_path);
		free(xf);
		return 1;
	}
	xf->bar0_cpu = bar_cpu;
	xf->bar0_gpu = bar_gpu;

	ret = xio::nvme_ep::createQueue(
		xf->controller_path, o->kmod_path ? o->kmod_path : ROCM_XIO_DEVICE_PATH,
		(uint16_t)o->queue_id, (uint16_t)o->queue_depth, xf->nvme_bdf,
		o->memory_mode, &xf->qinfo);
	if (ret < 0) {
		log_err("rocm-xio: createQueue: %s\n", strerror(-ret));
		free(xf->controller_path);
		free(xf);
		return 1;
	}
	xf->qinfo_valid = true;

	f->real_file_size = xf->nlba * (uint64_t)xf->lba_bytes;
	f->engine_data = xf;
	return 0;
}

static int fio_rocm_xio_close_file(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_options *o = eo(td);
	struct fio_rocm_xio_file *xf =
		static_cast<struct fio_rocm_xio_file *>(f->engine_data);
	int ctl_fd;

	(void)td;

	if (!xf)
		return 0;

	if (xf->data_buf_valid)
		xio::freeGpuAccessibleBuffer(&xf->data_buf);

	if (xf->qinfo_valid) {
		ctl_fd = open(xf->controller_path, O_RDWR);
		if (ctl_fd >= 0) {
			xio::nvme_ep::deleteQueue(ctl_fd, (uint16_t)o->queue_id);
			close(ctl_fd);
		}
	}

	free(xf->controller_path);
	free(xf);
	f->engine_data = nullptr;
	return 0;
}

static int fio_rocm_xio_get_file_size(struct thread_data *td,
				      struct fio_file *f)
{
	(void)td;
	(void)f;
	return 0;
}

static void fio_rocm_xio_terminate(struct thread_data *td)
{
	struct fio_rocm_xio_thread *tp = thread_priv(td);

	if (!tp)
		return;
	if (tp->prp_pool)
		free(tp->prp_pool);
	free(tp);
	td->io_ops_data = nullptr;
}

static enum fio_q_status fio_rocm_xio_queue(struct thread_data *td,
					    struct io_u *io_u)
{
	struct rocm_xio_options *o = eo(td);
	struct fio_rocm_xio_thread *tp = thread_priv(td);
	struct fio_rocm_xio_file *xf =
		static_cast<struct fio_rocm_xio_file *>(io_u->file->engine_data);
	unsigned mem_mode;
	void *xfer = io_u->buf;
	size_t buflen = io_u->xfer_buflen;
	uint64_t slba;
	uint32_t nlb;
	hipError_t he;
	int is_read;
	uint32_t prp_entries;
	uint32_t prp_stride;
	const char *kdev = o->kmod_path ? o->kmod_path : ROCM_XIO_DEVICE_PATH;

	fio_ro_check(td, io_u);

	if (!tp || !xf) {
		io_u->error = EINVAL;
		return FIO_Q_COMPLETED;
	}

	if (io_u->ddir == DDIR_TRIM || io_u->ddir == DDIR_SYNC ||
	    io_u->ddir == DDIR_DATASYNC) {
		io_u->error = EINVAL;
		return FIO_Q_COMPLETED;
	}

	if (io_u->ddir != DDIR_READ && io_u->ddir != DDIR_WRITE) {
		io_u->error = EINVAL;
		return FIO_Q_COMPLETED;
	}

	if (buflen % xf->lba_bytes || io_u->offset % xf->lba_bytes) {
		log_err("rocm-xio: offset and length must be LBA-aligned (%u)\n",
			xf->lba_bytes);
		io_u->error = EINVAL;
		return FIO_Q_COMPLETED;
	}

	slba = (uint64_t)(io_u->offset / xf->lba_bytes);
	nlb = (uint32_t)(buflen / xf->lba_bytes) - 1;

	if (o->use_device_mem)
		mem_mode = XIO_MEM_MODE_DATA_DEVICE;
	else
		mem_mode = 0;

	if (!xf->data_buf_valid || buflen > xf->data_buf_size) {
		if (xf->data_buf_valid) {
			xio::freeGpuAccessibleBuffer(&xf->data_buf);
			xf->data_buf_valid = false;
		}
		he = xio::allocateGpuAccessibleBuffer(buflen, mem_mode, xf->nvme_bdf,
						      xf->controller_path,
						      &xf->data_buf);
		if (he != hipSuccess) {
			log_err("rocm-xio: allocateGpuAccessibleBuffer failed\n");
			io_u->error = ENOMEM;
			return FIO_Q_COMPLETED;
		}
		xf->data_buf_valid = true;
		xf->data_buf_size = buflen;
	}

	is_read = (io_u->ddir == DDIR_READ);

	if (!is_read) {
		if (o->use_device_mem) {
			he = hipMemcpy(xf->data_buf.gpuPtr, xfer, buflen,
				       hipMemcpyDeviceToDevice);
		} else {
			he = hipMemcpy(xf->data_buf.gpuPtr, xfer, buflen,
				       hipMemcpyHostToDevice);
		}
		if (he != hipSuccess) {
			io_u->error = EIO;
			return FIO_Q_COMPLETED;
		}
	}

	xio::nvme_ep::nvmeDoorbellParams db = {};
	db.doorbellOffset = xio::nvme_ep::doorbellBase +
			    (uint32_t)o->queue_id * 2U *
				    xio::nvme_ep::doorBellStride;
	db.nvmeTargetBdf = xf->nvme_bdf;
	db.shadowBufferVirt = nullptr;
	db.nvmeBar0Gpu = xf->bar0_gpu;
	db.usePciMmioBridge = false;

	xio::nvme_ep::nvmeIoParams iop = {};
	iop.lbaSize = xf->lba_bytes;
	iop.baseLba = slba;
	iop.lbaRangeLbas = 0;
	iop.useRandomAccess = false;
	iop.readIo = is_read ? 1 : 0;
	iop.writeIo = is_read ? 0 : 1;
	iop.lfsrSeed = 0;
	iop.queueSize = o->queue_depth;
	iop.nsid = xf->nsid;
	iop.lbasPerIo = nlb + 1;
	iop.infiniteMode = false;
	iop.batchSize = 1;
	iop.wavefrontSize = 0;

	xio::nvme_ep::nvmeBufferParams bp = {};
	bp.bufferSize = xf->data_buf_size;
	if (is_read) {
		bp.readBuffer = static_cast<uint8_t *>(xf->data_buf.gpuPtr);
		bp.readBufferDma = xf->data_buf.dmaAddr;
		bp.readPagePhysAddrs = xf->data_buf.pagePhysAddrs;
		bp.readNumPages = xf->data_buf.numPages;
	} else {
		bp.writeBuffer = static_cast<uint8_t *>(xf->data_buf.gpuPtr);
		bp.writeBufferDma = xf->data_buf.dmaAddr;
		bp.writePagePhysAddrs = xf->data_buf.pagePhysAddrs;
		bp.writeNumPages = xf->data_buf.numPages;
	}

	prp_entries = (buflen > NVME_PAGE_SIZE)
			  ? (uint32_t)((buflen / NVME_PAGE_SIZE) + 1)
			  : 0;
	prp_stride = prp_entries;
	if (prp_stride > 1)
		prp_stride = next_pow2_u32(prp_stride);

	bp.prpEntriesPerCmd = prp_entries;
	bp.prpListPool = nullptr;
	bp.prpListPoolDma = 0;

	if (prp_entries > 0) {
		if (!tp->prp_pool || tp->prp_pool_bytes <
			(size_t)prp_stride * sizeof(uint64_t)) {
			io_u->error = EINVAL;
			return FIO_Q_COMPLETED;
		}
		memset(tp->prp_pool, 0, (size_t)prp_stride * sizeof(uint64_t));
		if (!tp->prp_pool_dma) {
			tp->prp_pool_dma =
				xio::getPhysAddr(tp->prp_pool, tp->prp_pool_bytes, false,
						 xf->nvme_bdf, kdev, false,
						 "fio rocm-xio prp");
			if (!tp->prp_pool_dma) {
				io_u->error = EIO;
				return FIO_Q_COMPLETED;
			}
		}
		bp.prpListPool = tp->prp_pool;
		bp.prpListPoolDma = tp->prp_pool_dma;
	}

	void *sq_ptr = xf->qinfo.sq_gpu ? xf->qinfo.sq_gpu : xf->qinfo.sq_virt;
	void *cq_ptr = xf->qinfo.cq_gpu ? xf->qinfo.cq_gpu : xf->qinfo.cq_virt;

	xio::XioEndpointConfig xcfg = {};
	xcfg.submissionQueue = sq_ptr;
	xcfg.completionQueue = cq_ptr;
	xcfg.memoryMode = o->memory_mode;
	xcfg.endpointConfig = nullptr;

	hipLaunchKernelGGL(xio::nvme_ep::gpuKernel, dim3(1), dim3(1), 0, 0, xcfg,
			   iop, db, bp);

	he = hipDeviceSynchronize();
	if (he != hipSuccess) {
		io_u->error = EIO;
		return FIO_Q_COMPLETED;
	}

	if (is_read) {
		if (o->use_device_mem) {
			he = hipMemcpy(xfer, xf->data_buf.gpuPtr, buflen,
				       hipMemcpyDeviceToDevice);
		} else {
			he = hipMemcpy(xfer, xf->data_buf.gpuPtr, buflen,
				       hipMemcpyDeviceToHost);
		}
		if (he != hipSuccess)
			io_u->error = EIO;
	}

	return FIO_Q_COMPLETED;
}

static void fio_rocm_xio_cleanup(struct thread_data *td)
{
	(void)td;
}

static struct fio_option options[] = {
	{
		.name = "rocm_xio_gpu",
		.lname = "ROCm XIO HIP device index",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, gpu_id),
		.help = "HIP device ID for GPU-initiated NVMe",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_queue_id",
		.lname = "ROCm XIO NVMe queue id",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, queue_id),
		.help = "NVMe I/O queue id (must not collide with other users)",
		.def = "1",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_queue_depth",
		.lname = "ROCm XIO NVMe queue depth",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, queue_depth),
		.help = "SQ/CQ depth (power of two, e.g. 64)",
		.def = "64",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_nsid",
		.lname = "ROCm XIO namespace id override",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, nsid),
		.help = "0=auto from /dev/nvmeXnY or NVME_IOCTL_ID",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_memory_mode",
		.lname = "ROCm XIO memory mode bits",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, memory_mode),
		.help = "Flags for createQueue (rocm-xio queue/data placement)",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_device_mem",
		.lname = "ROCm XIO use VRAM for DMA buffer",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, use_device_mem),
		.help = "1=device memory for I/O buffer (iomem cuda_malloc), 0=pinned host",
		.def = "1",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_kmod_path",
		.lname = "ROCm XIO kernel module device",
		.type = FIO_OPT_STR_STORE,
		.off1 = offsetof(struct rocm_xio_options, kmod_path),
		.help = "Path to rocm-xio char device (default " ROCM_XIO_DEVICE_PATH ")",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_controller",
		.lname = "ROCm XIO NVMe controller path override",
		.type = FIO_OPT_STR_STORE,
		.off1 = offsetof(struct rocm_xio_options, controller_override),
		.help = "e.g. /dev/nvme0 when filename is not a namespace path",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = NULL,
	},
};

static struct ioengine_ops ioengine = {
	.name = "rocm-xio",
	.version = FIO_IOOPS_VERSION,
	.flags = FIO_RAWIO | FIO_MEMALIGN | FIO_NOEXTEND,
	.setup = fio_rocm_xio_setup,
	.init = fio_rocm_xio_init,
	.post_init = fio_rocm_xio_post_init,
	.iomem_alloc = fio_rocm_xio_iomem_alloc,
	.queue = fio_rocm_xio_queue,
	.open_file = fio_rocm_xio_open_file,
	.close_file = fio_rocm_xio_close_file,
	.get_file_size = fio_rocm_xio_get_file_size,
	.cleanup = fio_rocm_xio_cleanup,
	.terminate = fio_rocm_xio_terminate,
	.option_struct_size = sizeof(struct rocm_xio_options),
	.options = options,
};

} // namespace

static void fio_init fio_rocm_xio_register(void)
{
	register_ioengine(&ioengine);
}

static void fio_exit fio_rocm_xio_unregister(void)
{
	unregister_ioengine(&ioengine);
}
