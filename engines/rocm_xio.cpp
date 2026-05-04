/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * rocm-xio I/O engine: NVMe read/write via GPU submission queues using
 * the ROCm rocm-xio library (AMD GPU-initiated NVMe I/O).
 */

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include <xio.h>
#include <endpoints/nvme-ep/nvme-ep.h>

extern "C" {
#include "../fio.h"
#include "../log.h"
#include "../optgroup.h"
}

#define GPU_ID_SEP ':'
#define ROCM_XIO_DEFAULT_KMOD "/dev/rocm-xio"

struct rocm_xio_options {
	struct thread_data *td;
	char *gpu_ids;
	char *kmod_device;
	char *controller_override;
	int my_gpu_id;
	unsigned int memory_mode;
	unsigned int queue_length;
	unsigned int nsid;
};

struct fio_rocm_xio_file_data {
	char *nvme_controller_path;
	int nvme_fd;
	uint16_t nvme_bdf;
	uint32_t lba_size;
	uint32_t nsid;
	uint16_t queue_id;
	struct nvme_queue_info qinfo;
	bool queue_created;
	void *bar0_cpu;
	void *bar0_gpu;
	uint16_t sq_tail;
	uint16_t cq_head;
	uint8_t cq_phase;
	uint32_t cmd_counter;
};

static struct fio_option options[] = {
	{
		.name = "gpu_dev_ids",
		.lname = "rocm-xio engine GPU device ids",
		.type = FIO_OPT_STR_STORE,
		.off1 = offsetof(struct rocm_xio_options, gpu_ids),
		.help = "HIP device indices for jobs with multiple processes, "
			"colon-separated (same convention as libcufile).",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_kmod",
		.lname = "rocm-xio kernel module device path",
		.type = FIO_OPT_STR_STORE,
		.off1 = offsetof(struct rocm_xio_options, kmod_device),
		.help = "Path to rocm-xio character device (default " ROCM_XIO_DEFAULT_KMOD ").",
		.def = ROCM_XIO_DEFAULT_KMOD,
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_controller",
		.lname = "NVMe controller path override",
		.type = FIO_OPT_STR_STORE,
		.off1 = offsetof(struct rocm_xio_options, controller_override),
		.help = "Block device jobs pass a namespace path (e.g. /dev/nvme0n1). "
			"Set this to the parent controller (e.g. /dev/nvme0) for queue "
			"setup. If unset, a /dev/nvmeXpY -> /dev/nvmeX heuristic is used.",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_memory_mode",
		.lname = "rocm-xio NVMe queue memory mode bitmask",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, memory_mode),
		.help = "Bits passed to rocm-xio createQueue (see rocm-xio XIO_MEM_MODE_*). "
			"Default 0 (host SQ/CQ).",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_queue_depth",
		.lname = "rocm-xio NVMe queue depth (entries)",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, queue_length),
		.help = "SQ/CQ depth (power of two). Must be at least iodepth+1.",
		.def = "64",
		.minval = 2,
		.maxval = 65536,
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_nsid",
		.lname = "rocm-xio NVMe namespace id",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, nsid),
		.help = "Namespace ID for read/write commands (default 1).",
		.def = "1",
		.minval = 1,
		.maxval = UINT_MAX,
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = NULL,
	},
};

static int rocm_xio_find_gpu_id(struct thread_data *td)
{
	struct rocm_xio_options *o =
		reinterpret_cast<struct rocm_xio_options *>(td->eo);
	int gpu_id = 0;

	if (o->gpu_ids != NULL) {
		char *gpu_ids, *pos, *cur;
		int i, id_count, gpu_idx;

		for (id_count = 0, cur = o->gpu_ids; cur != NULL; id_count++) {
			cur = strchr(cur, GPU_ID_SEP);
			if (cur != NULL)
				cur++;
		}

		gpu_idx = td->subjob_number % id_count;

		pos = gpu_ids = strdup(o->gpu_ids);
		if (gpu_ids == NULL) {
			log_err("rocm-xio: strdup(gpu_ids): errno=%d\n", errno);
			return -1;
		}

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

static char *rocm_xio_dup_controller_path(const char *filename,
					  const char *override)
{
	if (override != NULL && override[0] != '\0')
		return strdup(override);

	if (filename == NULL)
		return NULL;

	/* /dev/nvme0n1 -> /dev/nvme0 */
	const char *base = strrchr(filename, '/');
	const char *p = base ? base + 1 : filename;
	if (strncmp(p, "nvme", 4) != 0)
		return strdup(filename);

	const char *n = strchr(p + 4, 'n');
	if (n == NULL || n == p + 4)
		return strdup(filename);

	size_t ctl_len = (size_t)(n - filename);
	char *out = static_cast<char *>(malloc(ctl_len + 1));
	if (!out)
		return NULL;
	memcpy(out, filename, ctl_len);
	out[ctl_len] = '\0';
	return out;
}

static int rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o =
		reinterpret_cast<struct rocm_xio_options *>(td->eo);
	hipError_t h;

	o->my_gpu_id = rocm_xio_find_gpu_id(td);
	if (o->my_gpu_id < 0)
		return 1;

	h = hipSetDevice(o->my_gpu_id);
	if (h != hipSuccess) {
		log_err("rocm-xio: hipSetDevice(%d): %s\n", o->my_gpu_id,
			hipGetErrorString(h));
		return 1;
	}

	dprint(FD_MEM, "rocm-xio: subjob %d uses GPU %d\n", td->subjob_number,
	       o->my_gpu_id);
	return 0;
}

static void rocm_xio_destroy_queue(struct fio_rocm_xio_file_data *fd)
{
	if (!fd->queue_created)
		return;
	if (fd->nvme_fd >= 0)
		xio::nvme_ep::deleteQueue(fd->nvme_fd, fd->queue_id);
	fd->queue_created = false;
}

static int rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_options *o =
		reinterpret_cast<struct rocm_xio_options *>(td->eo);
	struct fio_rocm_xio_file_data *fd = NULL;
	const char *kmod = (o->kmod_device && o->kmod_device[0]) ? o->kmod_device
								  : ROCM_XIO_DEFAULT_KMOD;
	unsigned int qdepth = o->queue_length;
	int rc;

	rc = generic_open_file(td, f);
	if (rc)
		return rc;

	fd = static_cast<struct fio_rocm_xio_file_data *>(
		calloc(1, sizeof(*fd)));
	if (!fd) {
		rc = ENOMEM;
		goto err_close;
	}

	fd->nvme_fd = -1;
	fd->bar0_cpu = nullptr;
	fd->bar0_gpu = nullptr;
	fd->sq_tail = 0;
	fd->cq_head = 0;
	fd->cq_phase = 1;
	fd->cmd_counter = 0;
	fd->nvme_controller_path =
		rocm_xio_dup_controller_path(f->file_name, o->controller_override);
	if (!fd->nvme_controller_path) {
		log_err("rocm-xio: failed to allocate controller path\n");
		rc = ENOMEM;
		goto err_free;
	}

	fd->nvme_fd = open(fd->nvme_controller_path, O_RDWR);
	if (fd->nvme_fd < 0) {
		log_err("rocm-xio: open(%s): errno=%d\n", fd->nvme_controller_path,
			errno);
		rc = errno;
		goto err_free;
	}

	{
		unsigned lba_sz = 0;
		int qret = xio::nvme_ep::queryLbaSize(fd->nvme_controller_path,
						      o->nsid, &lba_sz);
		if (qret != 0) {
			log_err("rocm-xio: queryLbaSize(%s): %s\n",
				fd->nvme_controller_path, strerror(-qret));
			rc = EINVAL;
			goto err_free;
		}
		fd->lba_size = lba_sz;
	}

	{
		uint16_t bdf = 0;
		int bret = xio::detectBdfFromDevice(fd->nvme_controller_path, &bdf);
		if (bret != 0) {
			log_err("rocm-xio: detectBdfFromDevice: %s\n",
				strerror(-bret));
			rc = EINVAL;
			goto err_free;
		}
		fd->nvme_bdf = bdf;
	}

	fd->nsid = o->nsid;

	{
		uint16_t max_q = 0;
		int mq = xio::nvme_ep::queryMaxQueueId(fd->nvme_controller_path,
						       &max_q);
		if (mq != 0) {
			log_err("rocm-xio: queryMaxQueueId failed\n");
			rc = EINVAL;
			goto err_free;
		}
		/* One I/O queue per thread: map thread_number to distinct queue IDs. */
		uint32_t tid = td->thread_number;
		if (tid > (uint32_t)max_q) {
			log_err("rocm-xio: thread_number %u exceeds max queue id %u\n",
				tid, (unsigned)max_q);
			rc = EINVAL;
			goto err_free;
		}
		fd->queue_id = (uint16_t)(1 + (tid % (unsigned)max_q)));
	}

	if (qdepth < (unsigned int)td->o.iodepth + 1) {
		log_err("rocm-xio: rocm_xio_queue_depth=%u must be >= iodepth+1 (%u)\n",
			qdepth, td->o.iodepth + 1);
		rc = EINVAL;
		goto err_free;
	}

	(void)xio::nvme_ep::deleteQueue(fd->nvme_fd, fd->queue_id);

	{
		int cq = xio::nvme_ep::createQueue(fd->nvme_controller_path, kmod,
						   fd->queue_id, (uint16_t)qdepth,
						   fd->nvme_bdf, o->memory_mode,
						   &fd->qinfo);
		if (cq != 0) {
			log_err("rocm-xio: createQueue failed (%d)\n", cq);
			rc = EINVAL;
			goto err_free;
		}
	}
	fd->queue_created = true;

	if (xio::mapPciBar(fd->nvme_bdf, 0, &fd->bar0_cpu, &fd->bar0_gpu, 8192) !=
	    0) {
		log_err("rocm-xio: mapPciBar failed for NVMe BDF\n");
		rc = EINVAL;
		goto err_free;
	}

	FILE_SET_ENG_DATA(f, fd);
	return 0;

err_free:
	if (fd) {
		rocm_xio_destroy_queue(fd);
		if (fd->bar0_cpu) {
			munmap(fd->bar0_cpu, 8192);
			fd->bar0_cpu = nullptr;
			fd->bar0_gpu = nullptr;
		}
		if (fd->nvme_fd >= 0) {
			close(fd->nvme_fd);
			fd->nvme_fd = -1;
		}
		free(fd->nvme_controller_path);
		free(fd);
	}
err_close:
	generic_close_file(td, f);
	return rc;
}

static int rocm_xio_close_file(struct thread_data *td, struct fio_file *f)
{
	struct fio_rocm_xio_file_data *fd =
		static_cast<struct fio_rocm_xio_file_data *>(FILE_ENG_DATA(f));

	if (fd) {
		rocm_xio_destroy_queue(fd);
		if (fd->bar0_cpu)
			munmap(fd->bar0_cpu, 8192);
		if (fd->nvme_fd >= 0)
			close(fd->nvme_fd);
		free(fd->nvme_controller_path);
		free(fd);
		FILE_SET_ENG_DATA(f, NULL);
	}
	return generic_close_file(td, f);
}

static enum fio_q_status rocm_xio_queue(struct thread_data *td,
					struct io_u *io_u)
{
	struct rocm_xio_options *o =
		reinterpret_cast<struct rocm_xio_options *>(td->eo);
	struct fio_rocm_xio_file_data *fd =
		static_cast<struct fio_rocm_xio_file_data *>(
			FILE_ENG_DATA(io_u->file));
	volatile nvme_ep::sqeType *sqe_addr =
		static_cast<volatile nvme_ep::sqeType *>(fd->qinfo.sq_gpu);
	volatile nvme_ep::cqeType *cqe_addr =
		static_cast<volatile nvme_ep::cqeType *>(fd->qinfo.cq_gpu);
	const unsigned qs = o->queue_length;
	uint64_t slba;
	uint32_t nlb;
	nvme_ep::sqeType sqe = {};
	unsigned spins = 0;

	fio_ro_check(td, io_u);

	if (!fd || !sqe_addr || !cqe_addr) {
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "rocm-xio");
		return FIO_Q_COMPLETED;
	}

	switch (io_u->ddir) {
	case DDIR_READ:
	case DDIR_WRITE:
		break;
	case DDIR_SYNC:
	case DDIR_DATASYNC:
		if (fsync(io_u->file->fd) != 0) {
			io_u->error = errno;
			td_verror(td, io_u->error, "fsync");
		}
		return FIO_Q_COMPLETED;
	default:
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "rocm-xio ddir");
		return FIO_Q_COMPLETED;
	}

	if (io_u->xfer_buflen % fd->lba_size || io_u->offset % fd->lba_size) {
		io_u->error = EINVAL;
		log_err("rocm-xio: offset %llu or length %llu not LBA-aligned "
			"(lba_size=%u)\n",
			(unsigned long long)io_u->offset,
			(unsigned long long)io_u->xfer_buflen, fd->lba_size);
		td_verror(td, EINVAL, "rocm-xio align");
		return FIO_Q_COMPLETED;
	}

	slba = (uint64_t)(io_u->offset / fd->lba_size);
	nlb = (uint32_t)(io_u->xfer_buflen / fd->lba_size);

	sqe.opcode = (io_u->ddir == DDIR_READ) ? nvme_cmd_read : nvme_cmd_write;
	sqe.flags = 0;
	fd->cmd_counter++;
	sqe.command_id =
		(uint16_t)((fd->cmd_counter % 65535u) ? (fd->cmd_counter % 65535u)
						     : 1u);
	sqe.nsid = fd->nsid;

	{
		uint64_t dma = 0;
		hipError_t h = hipMemPtrGetInfo(io_u->xfer_buf, &dma);
		if (h != hipSuccess || dma == 0) {
			io_u->error = EINVAL;
			log_err("rocm-xio: buffer is not GPU device memory "
				"(use gpu_buffer=1)\n");
			td_verror(td, EINVAL, "rocm-xio buf");
			return FIO_Q_COMPLETED;
		}
		nvme_ep::calculatePrps(dma, (uint32_t)io_u->xfer_buflen, &sqe);
	}
	nvme_ep::sqeSetup(&sqe, slba, nlb);

	{
		const uint16_t sq_slot = fd->sq_tail % qs;
		const uint16_t cq_slot = fd->cq_head % qs;
		const uint8_t expected_phase = fd->cq_phase;
		nvme_ep::nvmeDoorbellParams db = {};
		db.doorbellOffset = nvme_ep::doorbellBase +
				    (uint32_t)fd->queue_id * nvme_ep::doorBellStride;
		db.nvmeTargetBdf = fd->nvme_bdf;
		db.shadowBufferVirt = nullptr;
		db.nvmeBar0Gpu = fd->bar0_gpu;
		db.usePciMmioBridge = false;

		nvme_ep::sqeWrite(sqe, &sqe_addr[sq_slot]);
		fd->sq_tail = (uint16_t)((sq_slot + 1) % qs);
		nvme_ep::ringDoorbell(fd->sq_tail, db);

		for (;;) {
			nvme_ep::cqeType cqe = nvme_ep::cqeRead(&cqe_addr[cq_slot]);
			uint8_t phase = NVME_CQE_STATUS_PHASE(cqe.status);
			(void)cqe;
			if (phase == expected_phase)
				break;
			spins++;
			if (spins >= nvme_ep::NVME_EP_MAX_POLLS) {
				io_u->error = ETIME;
				log_err("rocm-xio: CQ poll timeout\n");
				break;
			}
		}

		nvme_ep::cqeType cqe_new = nvme_ep::cqeRead(&cqe_addr[cq_slot]);
		if (!io_u->error && !nvme_ep::cqeOk(&cqe_new)) {
			io_u->error = EIO;
			log_err("rocm-xio: NVMe error sc=%u sct=%u\n",
				(unsigned)nvme_ep::cqeStatusCode(&cqe_new),
				(unsigned)nvme_ep::cqeStatusType(&cqe_new));
		}

		fd->cq_head = (uint16_t)((cq_slot + 1) % qs);
		if (fd->cq_head == 0)
			fd->cq_phase ^= 1u;
		nvme_ep::ringDoorbell(fd->cq_head, db,
				      db.doorbellOffset + nvme_ep::doorBellStride);
	}

	if (io_u->error)
		td_verror(td, io_u->error, "rocm-xio");

	return FIO_Q_COMPLETED;
}

static void rocm_xio_cleanup(struct thread_data *td)
{
	(void)td;
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name = "rocm-xio",
	.version = FIO_IOOPS_VERSION,
	.init = rocm_xio_init,
	.queue = rocm_xio_queue,
	.open_file = rocm_xio_open_file,
	.close_file = rocm_xio_close_file,
	.cleanup = rocm_xio_cleanup,
	.flags = FIO_SYNCIO | FIO_RAWIO | FIO_MEMALIGN,
	.options = options,
	.option_struct_size = sizeof(struct rocm_xio_options),
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
