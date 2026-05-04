// SPDX-License-Identifier: GPL-2.0
/*
 * rocm-xio I/O engine — NVMe read/write using AMD GPU-initiated I/O
 * (ROCm rocm-xio / nvme-ep). Requires the rocm-xio kernel module,
 * librocm-xio, HIP, and an NVMe namespace character device (/dev/ng*n*).
 */

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <climits>

#include <hip/hip_runtime.h>

#include <endpoints/nvme-ep/nvme-ep.h>
#include <xio.h>

extern "C" {
#include "../fio.h"
#include "../optgroup.h"
#include "nvme.h"
}

#define GPU_ID_SEP ":"

struct rocm_xio_options {
	struct thread_data *td;
	char *gpu_ids;
	int my_gpu_id;
	unsigned int queue_depth;
	unsigned int queue_id;
	unsigned int use_mmio_bridge;
};

struct rocm_xio_file_data {
	struct nvme_data nvme;
	uint16_t nvme_bdf;
	uint16_t qid;
	uint16_t qdepth;
	struct nvme_queue_info qi;
	xio::nvme_ep::nvmeDoorbellParams doorbell;
	bool use_mmio_bridge;
	void *bar0_cpu;
	void *shadow_host;
	uint16_t sq_tail;
	uint16_t cq_head;
	uint8_t cq_phase;
};

static struct fio_option options[] = {
	{
		.name		= "gpu_dev_ids",
		.lname		= "rocm-xio engine GPU device ids",
		.type		= FIO_OPT_STR_STORE,
		.off1		= offsetof(struct rocm_xio_options, gpu_ids),
		.help		= "HIP GPU indices (subjobs round-robin), separated by "
				  GPU_ID_SEP,
		.category	= FIO_OPT_C_ENGINE,
		.group		= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name		= "rocm_xio_queue_depth",
		.lname		= "rocm-xio NVMe queue depth",
		.type		= FIO_OPT_INT,
		.off1		= offsetof(struct rocm_xio_options, queue_depth),
		.def		= "64",
		.help		= "SQ/CQ depth (power of two)",
		.category	= FIO_OPT_C_ENGINE,
		.group		= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name		= "rocm_xio_queue_id",
		.lname		= "rocm-xio NVMe queue id",
		.type		= FIO_OPT_INT,
		.off1		= offsetof(struct rocm_xio_options, queue_id),
		.def		= "0",
		.help		= "NVMe I/O queue id (0 = assign from subjob index)",
		.category	= FIO_OPT_C_ENGINE,
		.group		= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name		= "rocm_xio_mmio_bridge",
		.lname		= "rocm-xio PCI MMIO bridge doorbells",
		.type		= FIO_OPT_BOOL,
		.off1		= offsetof(struct rocm_xio_options, use_mmio_bridge),
		.help		= "Route doorbells via PCI MMIO bridge (e.g. QEMU)",
		.category	= FIO_OPT_C_ENGINE,
		.group		= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name		= NULL,
	},
};

static int rocm_xio_pick_gpu(struct thread_data *td)
{
	struct rocm_xio_options *o = (struct rocm_xio_options *)td->eo;
	int gpu_id = 0;

	if (o->gpu_ids) {
		char *gpu_ids, *pos, *cur;
		int i, id_count, gpu_idx;

		for (id_count = 0, cur = o->gpu_ids; cur != NULL; id_count++) {
			cur = strchr(cur, GPU_ID_SEP[0]);
			if (cur)
				cur++;
		}

		gpu_idx = td->subjob_number % id_count;

		pos = gpu_ids = strdup(o->gpu_ids);
		if (!gpu_ids)
			return -1;

		i = 0;
		while (pos != NULL && i <= gpu_idx) {
			i++;
			cur = strsep(&pos, GPU_ID_SEP);
		}

		if (cur)
			gpu_id = atoi(cur);

		free(gpu_ids);
	}

	return gpu_id;
}

static int rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o = (struct rocm_xio_options *)td->eo;

	o->my_gpu_id = rocm_xio_pick_gpu(td);
	if (o->my_gpu_id < 0)
		return 1;

	if (hipSetDevice(o->my_gpu_id) != hipSuccess)
		return 1;

	if (!o->queue_depth)
		o->queue_depth = 64;

	return 0;
}

static void rocm_xio_cleanup(struct thread_data *td)
{
	(void)td;
}

static uint16_t rocm_xio_pick_qid(struct thread_data *td)
{
	struct rocm_xio_options *o = (struct rocm_xio_options *)td->eo;

	if (o->queue_id != 0)
		return (uint16_t)o->queue_id;

	return (uint16_t)(1 + (td->subjob_number % 31));
}

static int rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_options *o = (struct rocm_xio_options *)td->eo;
	struct rocm_xio_file_data *fd;
	__u64 nlba_dummy = 0;
	uint16_t bdf = 0;
	bool emulated = false;
	void *bar0_cpu = nullptr;
	void *bar0_gpu = nullptr;
	void *shadow_virt = nullptr;
	void *shadow_gpu = nullptr;
	int ret;

	ret = generic_open_file(td, f);
	if (ret)
		return ret;

	fd = (struct rocm_xio_file_data *)calloc(1, sizeof(*fd));
	if (!fd) {
		generic_close_file(td, f);
		return -ENOMEM;
	}

	ret = fio_nvme_get_info(f, &nlba_dummy, 0, &fd->nvme);
	if (ret) {
		free(fd);
		generic_close_file(td, f);
		return ret;
	}

	ret = xio::detectBdfFromDevice(f->file_name, &bdf);
	if (ret < 0) {
		log_err("rocm-xio: detectBdfFromDevice failed for %s\n",
			f->file_name);
		free(fd);
		generic_close_file(td, f);
		return -EINVAL;
	}

	(void)xio::detectEmulatedNvme(f->file_name, &emulated);

	fd->nvme_bdf = bdf;
	fd->qdepth = (uint16_t)o->queue_depth;
	fd->qid = rocm_xio_pick_qid(td);
	fd->use_mmio_bridge = o->use_mmio_bridge != 0;
	fd->cq_phase = 1;

	if (fd->use_mmio_bridge) {
		uint16_t bridge_bdf = 0;

		ret = xio::detectPciMmioBridgeBdf(&bridge_bdf);
		if (ret < 0) {
			log_err("rocm-xio: MMIO bridge required but not found\n");
			free(fd);
			generic_close_file(td, f);
			return -EINVAL;
		}
		ret = xio::mapMmioBridgeShadowBuffer(bridge_bdf, &shadow_virt);
		if (ret < 0) {
			log_err("rocm-xio: mapMmioBridgeShadowBuffer failed\n");
			free(fd);
			generic_close_file(td, f);
			return -EINVAL;
		}
		fd->shadow_host = shadow_virt;
		ret = xio::registerMmioBridgeShadowBufferForGpu(
			shadow_virt, 8192, &shadow_gpu);
		if (ret < 0) {
			log_err("rocm-xio: registerMmioBridgeShadowBufferForGpu "
				"failed\n");
			free(fd);
			generic_close_file(td, f);
			return -EINVAL;
		}
		fd->doorbell.shadowBufferVirt = shadow_gpu;
		fd->doorbell.nvmeBar0Gpu = nullptr;
	} else {
		ret = xio::mapPciBar(bdf, 0, &bar0_cpu, &bar0_gpu, 8192);
		if (ret < 0) {
			log_err("rocm-xio: mapPciBar(BAR0) failed (%d)\n", ret);
			free(fd);
			generic_close_file(td, f);
			return -EINVAL;
		}
		fd->bar0_cpu = bar0_cpu;
		fd->doorbell.nvmeBar0Gpu = bar0_gpu;
		fd->doorbell.shadowBufferVirt = nullptr;
	}

	fd->doorbell.usePciMmioBridge = fd->use_mmio_bridge;
	fd->doorbell.nvmeTargetBdf = bdf;
	fd->doorbell.doorbellOffset =
		xio::nvme_ep::doorbellBase +
		(uint32_t)(2 * fd->qid * xio::nvme_ep::doorBellStride);

	unsigned mem_mode = XIO_MEM_MODE_SQ_DEVICE | XIO_MEM_MODE_CQ_DEVICE |
			    XIO_MEM_MODE_DATA_DEVICE;

	ret = xio::nvme_ep::createQueue(f->file_name, ROCM_XIO_DEVICE_PATH,
					fd->qid, fd->qdepth, bdf, mem_mode,
					&fd->qi);
	if (ret < 0) {
		log_err("rocm-xio: createQueue failed: %s\n", strerror(-ret));
		if (fd->bar0_cpu)
			munmap(fd->bar0_cpu, 8192);
		free(fd);
		generic_close_file(td, f);
		return -EINVAL;
	}

	FILE_SET_ENG_DATA(f, fd);
	return 0;
}

static int rocm_xio_close_file(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_file_data *fd =
		(struct rocm_xio_file_data *)FILE_ENG_DATA(f);
	int nvme_fd;
	int rc;

	if (fd) {
		nvme_fd = open(f->file_name, O_RDWR);
		if (nvme_fd >= 0) {
			xio::nvme_ep::deleteQueue(nvme_fd, fd->qid);
			close(nvme_fd);
		}
		if (fd->bar0_cpu)
			munmap(fd->bar0_cpu, 8192);
		free(fd);
		FILE_SET_ENG_DATA(f, nullptr);
	}

	rc = generic_close_file(td, f);
	return rc;
}

static int rocm_xio_get_file_size(struct thread_data *td, struct fio_file *f)
{
	return generic_get_file_size(td, f);
}

/*
 * Fill PRP entries in sqe for an xfer to DMA address dma_base (full xfer).
 */
static void rocm_xio_fill_prps(uint64_t dma_base, uint32_t xfer,
			     xio::nvme_ep::sqeType *sqe,
			     uint64_t *prp_list_buf, uint64_t prp_list_dma,
			     uint64_t *page_phys, uint32_t num_phys_pages)
{
	uint64_t offset_in_page = dma_base & (NVME_PAGE_SIZE - 1);
	uint64_t first_page_size = NVME_PAGE_SIZE - offset_in_page;

	if (xfer <= first_page_size) {
		xio::nvme_ep::calculatePrps(dma_base, xfer, sqe);
		return;
	}

	uint32_t remaining = (uint32_t)(xfer - first_page_size);
	uint32_t num_remaining_pages =
		(remaining + NVME_PAGE_SIZE - 1) / NVME_PAGE_SIZE;

	constexpr uint32_t kMaxPrpListEntries = NVME_PAGE_SIZE / sizeof(uint64_t);

	if (num_remaining_pages <= kMaxPrpListEntries && prp_list_buf &&
	    prp_list_dma) {
		uint32_t page_off = 0;

		xio::nvme_ep::calculatePrps(dma_base, xfer, sqe, prp_list_buf,
					    prp_list_dma, page_phys, page_off);
	} else {
		xio::nvme_ep::calculatePrps(dma_base, xfer, sqe);
	}
}

static enum fio_q_status rocm_xio_queue(struct thread_data *td,
					struct io_u *io_u)
{
	struct rocm_xio_options *o = (struct rocm_xio_options *)td->eo;
	struct rocm_xio_file_data *fd =
		(struct rocm_xio_file_data *)FILE_ENG_DATA(io_u->file);
	xio::xioBufferInfo xbi = {};
	xio::nvme_ep::sqeType sqe = {};
	volatile xio::nvme_ep::sqeType *sqe_slot;
	volatile xio::nvme_ep::cqeType *cqe_slot;
	void *sq_base;
	void *cq_base;
	xio::nvme_ep::cqeType cqe_new;
	unsigned int mem_mode;
	uint64_t slba;
	uint32_t nlb;
	int is_write;
	uint32_t xfer = io_u->xfer_buflen;
	uint64_t *prp_host = nullptr;
	void *prp_gpu = nullptr;
	uint64_t prp_dma = 0;
	unsigned int polls;

	io_u->error = 0;

	if (hipSetDevice(o->my_gpu_id) != hipSuccess) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "hipSetDevice");
		return FIO_Q_COMPLETED;
	}

	if (io_u->ddir == DDIR_TRIM || io_u->ddir == DDIR_SYNC) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "rocm-xio");
		return FIO_Q_COMPLETED;
	}

	if (xfer % fd->nvme.lba_size) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "xfer_len");
		return FIO_Q_COMPLETED;
	}

	is_write = (io_u->ddir == DDIR_WRITE);
	mem_mode = XIO_MEM_MODE_SQ_DEVICE | XIO_MEM_MODE_CQ_DEVICE |
		   XIO_MEM_MODE_DATA_DEVICE;

	if (xio::allocateGpuAccessibleBuffer(xfer, mem_mode, fd->nvme_bdf,
					     io_u->file->file_name,
					     &xbi) != hipSuccess) {
		io_u->error = ENOMEM;
		td_verror(td, io_u->error, "allocateGpuAccessibleBuffer");
		return FIO_Q_COMPLETED;
	}

	void *xfer_dev =
		xbi.gpuPtr ? xbi.gpuPtr : static_cast<void *>(xbi.hostPtr);

	if (is_write &&
	    hipMemcpy(xfer_dev, io_u->xfer_buf, xfer,
		      hipMemcpyHostToDevice) != hipSuccess) {
		xio::freeGpuAccessibleBuffer(&xbi);
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "hipMemcpy");
		return FIO_Q_COMPLETED;
	}

	{
		uint64_t dma = xbi.dmaAddr;
		uint64_t off = dma & (NVME_PAGE_SIZE - 1);
		uint64_t first_page = NVME_PAGE_SIZE - off;
		uint32_t rem_pages;

		if (xfer <= first_page)
			rem_pages = 0;
		else {
			uint32_t rem = (uint32_t)(xfer - first_page);

			rem_pages = (rem + NVME_PAGE_SIZE - 1) / NVME_PAGE_SIZE;
		}

		if (rem_pages > 1) {
			if (hipHostMalloc(&prp_host, NVME_PAGE_SIZE,
					  hipHostMallocMapped) != hipSuccess) {
				xio::freeGpuAccessibleBuffer(&xbi);
				io_u->error = ENOMEM;
				td_verror(td, io_u->error, "hipHostMalloc prp");
				return FIO_Q_COMPLETED;
			}
			memset(prp_host, 0, NVME_PAGE_SIZE);
			prp_dma = xio::getPhysAddr(prp_host);
			if (!prp_dma) {
				(void)hipHostFree(prp_host);
				xio::freeGpuAccessibleBuffer(&xbi);
				io_u->error = EINVAL;
				td_verror(td, io_u->error, "prp phys");
				return FIO_Q_COMPLETED;
			}
			(void)hipHostGetDevicePointer(&prp_gpu, prp_host, 0);
			if (!prp_gpu)
				prp_gpu = prp_host;
		}
	}

	slba = (__u64)(io_u->offset >> fd->nvme.lba_shift);
	nlb = xfer / fd->nvme.lba_size;

	sqe.opcode = is_write ? nvme_cmd_write : nvme_cmd_read;
	sqe.flags = 0;
	sqe.command_id = (uint16_t)(((unsigned)fd->sq_tail % 65535u) + 1u);
	sqe.nsid = fd->nvme.nsid;

	rocm_xio_fill_prps(xbi.dmaAddr, xfer, &sqe,
			   reinterpret_cast<uint64_t *>(prp_gpu), prp_dma,
			   xbi.pagePhysAddrs, xbi.numPages);

	xio::nvme_ep::sqeSetup(&sqe, slba, nlb);

	sq_base = fd->qi.sq_gpu ? fd->qi.sq_gpu : fd->qi.sq_virt;
	cq_base = fd->qi.cq_gpu ? fd->qi.cq_gpu : fd->qi.cq_virt;

	sqe_slot = reinterpret_cast<volatile xio::nvme_ep::sqeType *>(
		static_cast<char *>(sq_base) +
		(size_t)fd->sq_tail * NVME_EP_SQE_SIZE);
	cqe_slot = reinterpret_cast<volatile xio::nvme_ep::cqeType *>(
		static_cast<char *>(cq_base) +
		(size_t)fd->cq_head * NVME_EP_CQE_SIZE);

	xio::nvme_ep::sqeWrite(sqe, sqe_slot);

	fd->sq_tail =
		(uint16_t)((fd->sq_tail + 1) % (unsigned)fd->qdepth);

	xio::nvme_ep::ringDoorbell(fd->sq_tail, fd->doorbell);

	for (polls = 0; polls < xio::nvme_ep::NVME_EP_MAX_POLLS; polls++) {
		cqe_new = xio::nvme_ep::cqeRead(cqe_slot);
		if (NVME_CQE_STATUS_PHASE(cqe_new.status) == fd->cq_phase)
			break;
	}

	if (polls >= xio::nvme_ep::NVME_EP_MAX_POLLS) {
		if (prp_host)
			(void)hipHostFree(prp_host);
		xio::freeGpuAccessibleBuffer(&xbi);
		io_u->error = ETIMEDOUT;
		td_verror(td, io_u->error, "cq_poll");
		return FIO_Q_COMPLETED;
	}

	if (!xio::nvme_ep::cqeOk(
		    reinterpret_cast<volatile xio::nvme_ep::cqeType *>(&cqe_new))) {
		if (prp_host)
			(void)hipHostFree(prp_host);
		xio::freeGpuAccessibleBuffer(&xbi);
		io_u->error = EIO;
		td_verror(td, io_u->error, "nvme_cqe");
		return FIO_Q_COMPLETED;
	}

	fd->cq_head =
		(uint16_t)((fd->cq_head + 1) % (unsigned)fd->qdepth);
	if (fd->cq_head == 0)
		fd->cq_phase ^= 1;

	xio::nvme_ep::ringDoorbell(fd->cq_head, fd->doorbell,
				   fd->doorbell.doorbellOffset +
					   xio::nvme_ep::doorBellStride);

	if (!is_write &&
	    hipMemcpy(io_u->xfer_buf, xfer_dev, xfer,
		      hipMemcpyDeviceToHost) != hipSuccess) {
		if (prp_host)
			(void)hipHostFree(prp_host);
		xio::freeGpuAccessibleBuffer(&xbi);
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "hipMemcpy readback");
		return FIO_Q_COMPLETED;
	}

	if (prp_host)
		(void)hipHostFree(prp_host);
	xio::freeGpuAccessibleBuffer(&xbi);
	return FIO_Q_COMPLETED;
}

static int rocm_xio_iomem_alloc(struct thread_data *td, size_t total_mem)
{
	struct rocm_xio_options *o = (struct rocm_xio_options *)td->eo;

	td->orig_buffer = calloc(1, total_mem);
	if (!td->orig_buffer)
		return 1;

	if (hipSetDevice(o->my_gpu_id) != hipSuccess)
		goto err;

	if (hipMalloc(&td->iobuf, total_mem) != hipSuccess)
		goto err;

	td->mmap_buf = td->iobuf;
	td->o.mem_type = MEM_CUDA_MALLOC;
	return 0;

err:
	free(td->orig_buffer);
	td->orig_buffer = nullptr;
	return 1;
}

static void rocm_xio_iomem_free(struct thread_data *td)
{
	if (td->iobuf) {
		(void)hipFree(td->iobuf);
		td->iobuf = nullptr;
		td->mmap_buf = nullptr;
	}
	if (td->orig_buffer) {
		free(td->orig_buffer);
		td->orig_buffer = nullptr;
	}
}

static struct ioengine_ops ioengine = {
	.name			= "rocm-xio",
	.version		= FIO_IOOPS_VERSION,
	.init			= rocm_xio_init,
	.queue			= rocm_xio_queue,
	.open_file		= rocm_xio_open_file,
	.close_file		= rocm_xio_close_file,
	.get_file_size		= rocm_xio_get_file_size,
	.iomem_alloc		= rocm_xio_iomem_alloc,
	.iomem_free		= rocm_xio_iomem_free,
	.cleanup		= rocm_xio_cleanup,
	.flags			= FIO_SYNCIO | FIO_RAWIO | FIO_MEMALIGN,
	.options		= options,
	.option_struct_size	= sizeof(struct rocm_xio_options),
};

extern "C" {

void fio_init fio_rocm_xio_register(void)
{
	register_ioengine(&ioengine);
}

void fio_exit fio_rocm_xio_unregister(void)
{
	unregister_ioengine(&ioengine);
}

}
