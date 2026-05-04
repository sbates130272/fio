/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * fio I/O engine using AMD ROCm rocm-xio for NVMe I/O with GPU-accessible
 * data buffers (DMA-BUF / VRAM registration). Submission and completion
 * queue handling follows the rocm-xio nvme-ep model on the host CPU.
 */

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

#include <hip/hip_runtime.h>

#include "../fio.h"
#include "../lib/pow2.h"
#include "../optgroup.h"

#include <xio.h>
#include <endpoints/nvme-ep/nvme-ep.h>

struct rocm_xio_options {
	struct thread_data *td;
	char *gpu_ids;
	int my_gpu_id;
	unsigned int memory_mode;
	unsigned int nsid;
	unsigned int queue_id;
	unsigned int queue_length;
	unsigned int batch_size;
	int pci_mmio_bridge;
	int mmio_bridge_bdf;
	int nvme_target_bdf;
};

struct rocm_xio_file_data {
	unsigned int lba_size;
	uint64_t capacity_lbas;
	bool queue_created;
	uint16_t nvme_qid;
	struct nvme_queue_info qi;
	void *mmio_shadow_cpu;
	void *mmio_shadow_gpu;
	void *nvme_bar0_cpu;
	void *nvme_bar0_gpu;
	bool mmio_mapped;
	bool bar_mapped;
};

struct rocm_xio_thread_data {
	void *gpu_buf;
	size_t gpu_buf_size;
	struct xioBufferInfo buf_info;
	unsigned int lba_size;
	bool buf_ready;
};

#define GPU_ID_SEP ":"

static pthread_mutex_t rocm_xio_once_lock = PTHREAD_MUTEX_INITIALIZER;
static int rocm_xio_env_done;

static int rocm_xio_find_gpu_id(struct thread_data *td)
{
	struct rocm_xio_options *o =
		static_cast<struct rocm_xio_options *>(td->eo);
	int gpu_id = 0;

	if (o->gpu_ids != NULL) {
		char *gpu_ids, *pos, *cur;
		int i, id_count, gpu_idx;

		for (id_count = 0, cur = o->gpu_ids; cur != NULL; id_count++) {
			cur = strchr(cur, GPU_ID_SEP[0]);
			if (cur != NULL)
				cur++;
		}

		gpu_idx = td->subjob_number % id_count;

		pos = gpu_ids = strdup(o->gpu_ids);
		if (gpu_ids == NULL) {
			log_err("strdup(gpu_ids): err=%d\n", errno);
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

static void rocm_xio_setup_env_once(void)
{
	pthread_mutex_lock(&rocm_xio_once_lock);
	if (!rocm_xio_env_done) {
		if (setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 0) != 0)
			log_info("rocm-xio: could not set HSA_FORCE_FINE_GRAIN_PCIE\n");
		rocm_xio_env_done = 1;
	}
	pthread_mutex_unlock(&rocm_xio_once_lock);
}

static int rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o =
		static_cast<struct rocm_xio_options *>(td->eo);
	hipError_t h;

	rocm_xio_setup_env_once();

	o->my_gpu_id = rocm_xio_find_gpu_id(td);
	if (o->my_gpu_id < 0)
		return 1;

	h = hipSetDevice(o->my_gpu_id);
	if (h != hipSuccess) {
		log_err("hipSetDevice(%d): %s\n", o->my_gpu_id,
			hipGetErrorString(h));
		return 1;
	}

	if (!xio::checkKernelModuleLoaded()) {
		log_err("rocm-xio: kernel module not loaded\n");
		return 1;
	}

	dprint(FD_MEM, "rocm-xio: subjob %d uses GPU %d\n", td->subjob_number,
	       o->my_gpu_id);
	return 0;
}

static int rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_options *o =
		static_cast<struct rocm_xio_options *>(td->eo);
	struct rocm_xio_file_data *fdat;
	xio::nvme_ep::nvmeEpConfig probe;
	unsigned int lba = 0;
	uint64_t cap = 0;
	int rc;

	rc = generic_open_file(td, f);
	if (rc)
		return rc;

	fdat = static_cast<struct rocm_xio_file_data *>(calloc(1, sizeof(*fdat)));
	if (!fdat) {
		generic_close_file(td, f);
		return -ENOMEM;
	}

	if (xio::nvme_ep::queryLbaSize(f->file_name, o->nsid, &lba) != 0) {
		log_err("rocm-xio: queryLbaSize failed for %s\n", f->file_name);
		free(fdat);
		generic_close_file(td, f);
		return -EINVAL;
	}
	if (lba == 0 || (lba & (lba - 1)) != 0) {
		log_err("rocm-xio: invalid LBA size %u for %s\n", lba,
			f->file_name);
		free(fdat);
		generic_close_file(td, f);
		return -EINVAL;
	}

	if (xio::nvme_ep::queryNamespaceCapacity(f->file_name, o->nsid, &cap) !=
		    0 ||
	    cap == 0) {
		log_err("rocm-xio: queryNamespaceCapacity failed for %s\n",
			f->file_name);
		free(fdat);
		generic_close_file(td, f);
		return -EINVAL;
	}

	fdat->lba_size = lba;
	fdat->capacity_lbas = cap;

	probe.controller = f->file_name;
	probe.queueId = (uint16_t)o->queue_id;
	if (probe.queueId == 0) {
		uint16_t max_q = 0;

		if (xio::nvme_ep::queryMaxQueueId(f->file_name, &max_q) == 0 &&
		    max_q > 0)
			probe.queueId = max_q;
		else
			probe.queueId = 1;
	}
	probe.queueLength = (uint16_t)o->queue_length;
	probe.numQueues = 1;
	probe.ioParams.nsid = o->nsid;
	probe.ioParams.readIo = 1;
	probe.ioParams.writeIo = 0;
	probe.ioParams.batchSize = 1;
	probe.doorbellParams.usePciMmioBridge = o->pci_mmio_bridge != 0;
	probe.doorbellParams.mmioBridgeBdf =
		o->mmio_bridge_bdf ? (uint16_t)o->mmio_bridge_bdf : 0x0020;
	probe.doorbellParams.nvmeTargetBdf =
		o->nvme_target_bdf ? (uint16_t)o->nvme_target_bdf : 0;

	{
		std::string err = xio::nvme_ep::validateConfig(&probe);

		if (!err.empty()) {
			log_err("rocm-xio: %s\n", err.c_str());
			free(fdat);
			generic_close_file(td, f);
			return -EINVAL;
		}
	}

	if (probe.doorbellParams.usePciMmioBridge) {
		void *cpu = nullptr;

		rc = xio::mapMmioBridgeShadowBuffer(
			probe.doorbellParams.mmioBridgeBdf, &cpu);
		if (rc < 0) {
			log_err("rocm-xio: mapMmioBridgeShadowBuffer: %s\n",
				strerror(-rc));
			free(fdat);
			generic_close_file(td, f);
			return -EINVAL;
		}
		void *gpu = nullptr;
		rc = xio::registerMmioBridgeShadowBufferForGpu(cpu, 8192, &gpu);
		if (rc < 0) {
			log_err("rocm-xio: registerMmioBridgeShadowBufferForGpu: "
				"%s\n",
				strerror(-rc));
			free(fdat);
			generic_close_file(td, f);
			return -EINVAL;
		}
		fdat->mmio_shadow_cpu = cpu;
		fdat->mmio_shadow_gpu = gpu;
		fdat->mmio_mapped = true;
	} else {
		void *cpu = nullptr;
		void *gpu = nullptr;

		rc = xio::mapPciBar(probe.doorbellParams.nvmeTargetBdf, 0, &cpu,
				    &gpu, 8192);
		if (rc < 0) {
			log_err("rocm-xio: mapPciBar: %s\n", strerror(-rc));
			free(fdat);
			generic_close_file(td, f);
			return -EINVAL;
		}
		fdat->nvme_bar0_cpu = cpu;
		fdat->nvme_bar0_gpu = gpu;
		fdat->bar_mapped = true;
	}

	rc = xio::nvme_ep::createQueue(f->file_name, ROCM_XIO_DEVICE_PATH,
				       probe.queueId, probe.queueLength,
				       probe.doorbellParams.nvmeTargetBdf,
				       o->memory_mode, &fdat->qi);
	if (rc < 0) {
		log_err("rocm-xio: createQueue(qid=%u) failed: %s\n",
			(unsigned)probe.queueId, strerror(-rc));
		free(fdat);
		generic_close_file(td, f);
		return -EINVAL;
	}

	fdat->queue_created = true;
	fdat->nvme_qid = probe.queueId;
	FILE_SET_ENG_DATA(f, fdat);
	return 0;
}

static int rocm_xio_close_file(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_file_data *fdat = static_cast<struct rocm_xio_file_data *>(
		FILE_ENG_DATA(f));
	int rc;

	if (fdat && fdat->queue_created) {
		xio::nvme_ep::nvmeEpConfig cleanup_cfg;

		cleanup_cfg.controller = f->file_name;
		cleanup_cfg.queuesCreated = true;
		cleanup_cfg.queueIds.clear();
		cleanup_cfg.queueIds.push_back(fdat->nvme_qid);
		cleanup_cfg.queueInfos.clear();
		cleanup_cfg.queueInfos.push_back(fdat->qi);
		(void)xio::nvme_ep::nvme_ep_cleanup_queues(&cleanup_cfg);
		fdat->queue_created = false;
	}

	if (fdat) {
		if (fdat->mmio_mapped && fdat->mmio_shadow_cpu)
			(void)munmap(fdat->mmio_shadow_cpu, 8192);
		if (fdat->bar_mapped && fdat->nvme_bar0_cpu)
			(void)munmap(fdat->nvme_bar0_cpu, 8192);
		FILE_SET_ENG_DATA(f, NULL);
		free(fdat);
	}

	rc = generic_close_file(td, f);
	return rc;
}

static int rocm_xio_iomem_alloc(struct thread_data *td, size_t total_mem)
{
	struct rocm_xio_options *o =
		static_cast<struct rocm_xio_options *>(td->eo);
	struct rocm_xio_thread_data *priv;
	hipError_t h;

	priv = static_cast<struct rocm_xio_thread_data *>(calloc(1, sizeof(*priv)));
	if (!priv)
		return 1;

	td->io_ops_data = priv;

	h = hipSetDevice(o->my_gpu_id);
	if (h != hipSuccess) {
		log_err("hipSetDevice: %s\n", hipGetErrorString(h));
		goto err;
	}

	priv->gpu_buf_size = total_mem;
	h = hipMalloc(&priv->gpu_buf, total_mem);
	if (h != hipSuccess) {
		log_err("hipMalloc(%zu): %s\n", total_mem, hipGetErrorString(h));
		goto err;
	}

	memset(&priv->buf_info, 0, sizeof(priv->buf_info));
	priv->buf_ready = false;
	priv->lba_size = 0;

	td->orig_buffer = calloc(1, total_mem);
	if (!td->orig_buffer) {
		log_err("orig_buffer calloc failed: err=%d\n", errno);
		goto err;
	}

	return 0;

err:
	if (priv->gpu_buf) {
		(void)hipFree(priv->gpu_buf);
		priv->gpu_buf = nullptr;
	}
	free(priv);
	td->io_ops_data = NULL;
	return 1;
}

static void rocm_xio_iomem_free(struct thread_data *td)
{
	struct rocm_xio_thread_data *priv =
		static_cast<struct rocm_xio_thread_data *>(td->io_ops_data);

	if (!priv)
		return;

	if (priv->buf_ready)
		xio::freeGpuAccessibleBuffer(&priv->buf_info);

	if (priv->gpu_buf) {
		(void)hipFree(priv->gpu_buf);
		priv->gpu_buf = nullptr;
	}

	free(priv);
	td->io_ops_data = NULL;

	if (td->orig_buffer) {
		free(td->orig_buffer);
		td->orig_buffer = NULL;
	}
}

static int rocm_xio_ensure_buf(struct thread_data *td, struct fio_file *f,
			       unsigned int lba_size)
{
	struct rocm_xio_options *o =
		static_cast<struct rocm_xio_options *>(td->eo);
	struct rocm_xio_thread_data *priv =
		static_cast<struct rocm_xio_thread_data *>(td->io_ops_data);
	uint16_t nvme_bdf = 0;
	bool emulated = false;
	hipError_t h;

	if (priv->buf_ready && priv->lba_size == lba_size)
		return 0;

	if (priv->buf_ready) {
		xio::freeGpuAccessibleBuffer(&priv->buf_info);
		memset(&priv->buf_info, 0, sizeof(priv->buf_info));
		priv->buf_ready = false;
	}

	if (xio::detectBdfFromDevice(f->file_name, &nvme_bdf) != 0) {
		log_err("rocm-xio: detectBdfFromDevice failed for %s\n",
			f->file_name);
		return -EINVAL;
	}
	if (xio::detectEmulatedNvme(f->file_name, &emulated) != 0) {
		log_err("rocm-xio: detectEmulatedNvme failed for %s\n",
			f->file_name);
		return -EINVAL;
	}

	h = hipSetDevice(o->my_gpu_id);
	if (h != hipSuccess)
		return -EINVAL;

	h = xio::allocateGpuAccessibleBuffer(priv->gpu_buf_size, o->memory_mode,
					     nvme_bdf, f->file_name,
					     &priv->buf_info);
	if (h != hipSuccess) {
		log_err("rocm-xio: allocateGpuAccessibleBuffer: %s\n",
			hipGetErrorString(h));
		return -EINVAL;
	}

	priv->buf_ready = true;
	priv->lba_size = lba_size;
	return 0;
}

namespace {

static uint64_t rocm_xio_get_lba(unsigned op_index, uint16_t cmd_id,
			       uint32_t blocks,
			       const xio::nvme_ep::nvmeIoParams &ioParams)
{
	if (ioParams.useRandomAccess && ioParams.lbaRangeLbas > 0) {
		uint64_t seed = (uint64_t)op_index * 0x9e3779b9 +
				(uint64_t)cmd_id * 0x85ebca6b +
				(uint64_t)ioParams.lfsrSeed;
		seed ^= seed >> 16;
		seed *= 0xc2b2ae35;
		seed ^= seed >> 13;
		seed *= 0x9e3779b9;
		seed ^= seed >> 16;
		uint64_t max_lba =
			(blocks > 0 && ioParams.lbaRangeLbas >= blocks)
				? (static_cast<uint64_t>(ioParams.lbaRangeLbas) -
				   static_cast<uint64_t>(blocks) + 1U)
				: 1;
		return ioParams.baseLba + (seed % max_lba);
	}
	if (ioParams.lbaRangeLbas > 0) {
		return ioParams.baseLba +
		       ((op_index * blocks) % ioParams.lbaRangeLbas);
	}
	return ioParams.baseLba + (op_index * blocks);
}

/*
 * Host-side single-thread NVMe burst matching nvme-ep driveEndpointSingle
 * for batch_size <= 1, without GPU kernels.
 */
static hipError_t rocm_xio_run_host_burst(
	const xio::XioEndpointConfig &cfg_in,
	xio::nvme_ep::nvmeEpConfig *nvme_cfg,
	const xio::nvme_ep::nvmeBufferParams &buffer_params,
	const xio::nvme_ep::nvmeIoParams &io_params_in,
	xio::nvme_ep::nvmeDoorbellParams doorbell_params)
{
	using xio::nvme_ep::cqeRead;
	using xio::nvme_ep::cqeOk;
	using xio::nvme_ep::cqeStatusCode;
	using xio::nvme_ep::cqeStatusType;
	using xio::nvme_ep::NVME_CQE_STATUS_PHASE;
	using xio::nvme_ep::NVME_EP_MAX_POLLS;
	using xio::nvme_ep::calculatePrps;
	using xio::nvme_ep::nvme_cmd_read;
	using xio::nvme_ep::nvme_cmd_write;
	using xio::nvme_ep::ringDoorbell;
	using xio::nvme_ep::sqeSetup;
	using xio::nvme_ep::sqeWrite;
	using xio::nvme_ep::sqeType;
	using xio::nvme_ep::cqeType;

	xio::XioEndpointConfig config = cfg_in;
	config.endpointConfig = nullptr;

	void *sq_ptr = nvme_cfg->queueInfo.sq_gpu
			       ? nvme_cfg->queueInfo.sq_gpu
			       : nvme_cfg->queueInfo.sq_virt;
	void *cq_ptr = nvme_cfg->queueInfo.cq_gpu
			       ? nvme_cfg->queueInfo.cq_gpu
			       : nvme_cfg->queueInfo.cq_virt;
	if (!sq_ptr || !cq_ptr)
		return hipErrorInvalidValue;

	config.submissionQueue = sq_ptr;
	config.completionQueue = cq_ptr;

	sqeType *sqe_addr = static_cast<sqeType *>(config.submissionQueue);
	cqeType *cqe_addr = static_cast<cqeType *>(config.completionQueue);

	xio::nvme_ep::nvmeIoParams io_params = io_params_in;

	const unsigned num_read_ops = (unsigned)io_params.readIo;
	const unsigned num_write_ops = (unsigned)io_params.writeIo;
	const unsigned total_ops = num_read_ops + num_write_ops;

	uint32_t batch_size = io_params.batchSize;
	if (batch_size == 0)
		batch_size = total_ops;
	if (batch_size > (uint32_t)(io_params.queueSize - 1))
		batch_size = io_params.queueSize - 1;
	if (!io_params.infiniteMode && batch_size > total_ops)
		batch_size = total_ops;

	const uint32_t nsid = io_params.nsid;
	const uint32_t blocks = io_params.lbasPerIo;
	const size_t transfer_size = (size_t)blocks * io_params.lbaSize;

	const bool has_write = (buffer_params.writeBuffer != nullptr &&
				buffer_params.bufferSize > 0);
	const bool has_read = (buffer_params.readBuffer != nullptr &&
			       buffer_params.bufferSize > 0);

	if (has_write) {
		xio::DataPatternParams pp{
			buffer_params.writeBuffer, buffer_params.bufferSize, 0,
			io_params.lbaSize, io_params.lfsrSeed, nullptr};
		xio::dataPattern(false, pp);
	}

	uint16_t cq_head = 0;
	uint8_t expected_phase = 1;
	uint16_t sq_tail = 0;

	unsigned ops_done = 0;
	while ((io_params.infiniteMode || ops_done < total_ops) &&
	       (config.stopRequested == nullptr || !*config.stopRequested)) {
		unsigned batch_count = batch_size;
		if (!io_params.infiniteMode &&
		    ops_done + batch_count > total_ops)
			batch_count = total_ops - ops_done;

		for (unsigned b = 0; b < batch_count; b++) {
			unsigned op_idx = ops_done + b;
			uint16_t cmd_id =
				(uint16_t)((op_idx % 65535) + 1);

			bool is_read_op;
			if (io_params.infiniteMode) {
				is_read_op = (num_read_ops > 0);
			} else if (num_write_ops > 0 && num_read_ops > 0) {
				is_read_op = (op_idx >= num_write_ops);
			} else {
				is_read_op = (num_read_ops > 0);
			}

			unsigned lba_idx = op_idx;
			if (is_read_op && num_write_ops > 0)
				lba_idx = op_idx - num_write_ops;
			uint16_t lba_cmd_id =
				(uint16_t)((lba_idx % 65535) + 1);
			uint64_t lba =
				rocm_xio_get_lba(lba_idx, lba_cmd_id, blocks,
						 io_params);

			sqeType sqe_local = {};
			sqeType *sqe = &sqe_local;
			sqe->opcode = is_read_op ? nvme_cmd_read : nvme_cmd_write;
			sqe->command_id = cmd_id;
			sqe->nsid = nsid;

			{
				uint64_t buf_off = (uint64_t)b * transfer_size;
				uint32_t prp_off = b * buffer_params.prpEntriesPerCmd;
				uint64_t *prp_slot =
					buffer_params.prpListPool
						? buffer_params.prpListPool +
							  prp_off
						: nullptr;
				uint64_t prp_slot_dma =
					buffer_params.prpListPoolDma
						? buffer_params.prpListPoolDma +
							  prp_off * sizeof(uint64_t)
						: 0;
				uint32_t page_off =
					(uint32_t)(buf_off /
						   xio::nvme_ep::NVME_PAGE_SIZE);
				uint64_t intra_page_off =
					buf_off % xio::nvme_ep::NVME_PAGE_SIZE;

				if (is_read_op && has_read) {
					if (buf_off + transfer_size >
					    buffer_params.bufferSize)
						return hipErrorInvalidValue;
					uint64_t addr =
						buffer_params.readPagePhysAddrs
							? buffer_params
								  .readPagePhysAddrs
									  [page_off] +
								  intra_page_off
							: buffer_params.readBufferDma
								? (buffer_params
									   .readBufferDma +
								   buf_off)
								: (uint64_t)(
									buffer_params
										.readBuffer +
									buf_off);
					calculatePrps(
						addr, (uint32_t)transfer_size,
						sqe, prp_slot, prp_slot_dma,
						buffer_params.readPagePhysAddrs,
						page_off);
				} else if (!is_read_op && has_write) {
					if (buf_off + transfer_size >
					    buffer_params.bufferSize)
						return hipErrorInvalidValue;
					uint64_t addr =
						buffer_params.writePagePhysAddrs
							? buffer_params
								  .writePagePhysAddrs
									  [page_off] +
								  intra_page_off
							: buffer_params.writeBufferDma
								? (buffer_params
									   .writeBufferDma +
								   buf_off)
								: (uint64_t)(
									buffer_params
										.writeBuffer +
									buf_off);
					calculatePrps(
						addr, (uint32_t)transfer_size,
						sqe, prp_slot, prp_slot_dma,
						buffer_params.writePagePhysAddrs,
						page_off);
				} else {
					return hipErrorInvalidValue;
				}
			}

			sqeSetup(sqe, lba, blocks);

			sqeWrite(sqe_local, &sqe_addr[sq_tail]);
			sq_tail = (uint16_t)((sq_tail + 1) % io_params.queueSize);
		}

		ringDoorbell(sq_tail, doorbell_params);

		for (unsigned b = 0; b < batch_count; b++) {
			(void)b;
			volatile cqeType *cqe_entry = &cqe_addr[cq_head];

			cqeType cqe_new;
			unsigned poll_count = 0;
			while (true) {
				if (config.stopRequested != nullptr &&
				    poll_count % 100 == 0 &&
				    *config.stopRequested)
					return hipErrorNotSupported;
				cqe_new = cqeRead(cqe_entry);
				uint8_t phase =
					NVME_CQE_STATUS_PHASE(cqe_new.status);
				if (phase == expected_phase)
					break;
				poll_count++;
				if (poll_count >= NVME_EP_MAX_POLLS) {
					log_err("rocm-xio: CQ poll timeout\n");
					return hipErrorTimeout;
				}
			}

			if (!cqeOk(&cqe_new)) {
				log_err("rocm-xio: CQE error status=0x%04x "
					"sc=0x%02x sct=0x%02x\n",
					(unsigned)cqe_new.status,
					(unsigned)cqeStatusCode(&cqe_new),
					(unsigned)cqeStatusType(&cqe_new));
				return hipErrorUnknown;
			}

			cq_head = (uint16_t)((cq_head + 1) % io_params.queueSize);
			if (cq_head == 0)
				expected_phase ^= 1;
		}

		ringDoorbell(cq_head, doorbell_params,
			     doorbell_params.doorbellOffset +
				     xio::nvme_ep::doorBellStride);

		ops_done += batch_count;
	}

	return hipSuccess;
}

static hipError_t rocm_xio_submit_one(struct thread_data *td,
				      struct fio_file *f, enum fio_ddir ddir,
				      uint64_t io_offset, void *host_ptr,
				      unsigned long long buflen)
{
	struct rocm_xio_options *o =
		static_cast<struct rocm_xio_options *>(td->eo);
	struct rocm_xio_file_data *fdat = static_cast<struct rocm_xio_file_data *>(
		FILE_ENG_DATA(f));
	struct rocm_xio_thread_data *priv =
		static_cast<struct rocm_xio_thread_data *>(td->io_ops_data);
	unsigned int lba_size = fdat->lba_size;
	uint64_t slba;
	uint32_t nlb;
	uint32_t eff_batch;
	uint64_t xfer;
	uint32_t prp_stride_per_cmd;
	uint32_t prp_pool_entries;
	size_t prp_pool_bytes;
	long sys_page_sz;
	size_t host_page;
	hipError_t h;
	int ret;

	if (!fdat->queue_created)
		return hipErrorInvalidValue;

	if (buflen % lba_size != 0 || io_offset % lba_size != 0) {
		log_err("rocm-xio: offset %llu / len %llu not LBA-aligned "
			"(%u)\n",
			(unsigned long long)io_offset,
			(unsigned long long)buflen, lba_size);
		return hipErrorInvalidValue;
	}

	slba = io_offset / lba_size;
	nlb = (uint32_t)(buflen / lba_size);
	if (nlb == 0 || slba + nlb > fdat->capacity_lbas)
		return hipErrorInvalidValue;

	ret = rocm_xio_ensure_buf(td, f, lba_size);
	if (ret < 0)
		return hipErrorInvalidValue;

	h = hipSetDevice(o->my_gpu_id);
	if (h != hipSuccess)
		return h;

	if (ddir == DDIR_WRITE) {
		h = hipMemcpy(priv->gpu_buf, host_ptr, (size_t)buflen,
			      hipMemcpyHostToDevice);
		if (h != hipSuccess)
			return h;
		h = hipMemcpy(priv->buf_info.gpuPtr ? priv->buf_info.gpuPtr
						    : priv->buf_info.hostPtr,
			      priv->gpu_buf, (size_t)buflen, hipMemcpyDefault);
		if (h != hipSuccess)
			return h;
	}

	eff_batch = o->batch_size;
	if (eff_batch == 0)
		eff_batch = 1;
	if (eff_batch > (uint32_t)(o->queue_length - 1))
		eff_batch = o->queue_length - 1;

	xfer = (uint64_t)nlb * lba_size;
	if ((uint64_t)eff_batch * xfer > priv->gpu_buf_size)
		return hipErrorInvalidValue;

	prp_stride_per_cmd =
		(xfer > xio::nvme_ep::NVME_PAGE_SIZE)
			? (uint32_t)((xfer / xio::nvme_ep::NVME_PAGE_SIZE) + 1)
			: 0;
	if (prp_stride_per_cmd > 0) {
		uint32_t s = 1;
		while (s < prp_stride_per_cmd)
			s <<= 1;
		prp_stride_per_cmd = s;
	}
	prp_pool_entries = prp_stride_per_cmd * eff_batch;
	prp_pool_bytes =
		(size_t)prp_pool_entries * sizeof(uint64_t);
	sys_page_sz = sysconf(_SC_PAGESIZE);
	host_page = (sys_page_sz > 0) ? (size_t)sys_page_sz : 4096;
	if (prp_pool_bytes > 0)
		prp_pool_bytes =
			(prp_pool_bytes + host_page - 1) & ~(host_page - 1);

	void *prp_pool = nullptr;
	void *prp_gpu = nullptr;
	uint64_t prp_dma = 0;

	if (prp_pool_bytes > 0) {
		hipError_t pe = hipHostMalloc(&prp_pool, prp_pool_bytes,
					       hipHostMallocMapped);
		if (pe != hipSuccess || !prp_pool)
			return hipErrorMemoryAllocation;
		memset(prp_pool, 0, prp_pool_bytes);
		prp_dma = xio::getPhysAddr(prp_pool);
		if (prp_dma == 0) {
			(void)hipHostFree(prp_pool);
			return hipErrorInvalidValue;
		}
		(void)hipHostGetDevicePointer(&prp_gpu, prp_pool, 0);
		if (!prp_gpu)
			prp_gpu = prp_pool;
	}

	xio::nvme_ep::nvmeBufferParams bp = {};
	if (ddir == DDIR_READ) {
		bp.readBuffer = static_cast<uint8_t *>(
			priv->buf_info.gpuPtr ? priv->buf_info.gpuPtr
					      : priv->buf_info.hostPtr);
		bp.readBufferDma = priv->buf_info.dmaAddr;
		bp.readPagePhysAddrs = priv->buf_info.pagePhysAddrs;
		bp.readNumPages = priv->buf_info.numPages;
	} else {
		bp.writeBuffer = static_cast<uint8_t *>(
			priv->buf_info.gpuPtr ? priv->buf_info.gpuPtr
					       : priv->buf_info.hostPtr);
		bp.writeBufferDma = priv->buf_info.dmaAddr;
		bp.writePagePhysAddrs = priv->buf_info.pagePhysAddrs;
		bp.writeNumPages = priv->buf_info.numPages;
	}
	bp.bufferSize = priv->gpu_buf_size;
	bp.prpListPool =
		prp_gpu ? static_cast<uint64_t *>(prp_gpu) : nullptr;
	bp.prpListPoolDma = prp_dma;
	bp.prpEntriesPerCmd = prp_stride_per_cmd;

	xio::nvme_ep::nvmeIoParams iop = {};
	iop.lbaSize = lba_size;
	iop.baseLba = slba;
	iop.lbaRangeLbas = 0;
	iop.useRandomAccess = false;
	iop.readIo = (ddir == DDIR_READ) ? 1 : 0;
	iop.writeIo = (ddir == DDIR_WRITE) ? 1 : 0;
	iop.lfsrSeed = 0;
	iop.nsid = o->nsid;
	iop.lbasPerIo = nlb;
	iop.queueSize = (uint16_t)o->queue_length;
	iop.infiniteMode = false;
	iop.batchSize = o->batch_size;
	iop.wavefrontSize = 0;

	xio::nvme_ep::nvmeEpConfig ncfg = {};
	ncfg.queueInfo = fdat->qi;

	xio::XioEndpointConfig xcfg;
	xcfg.numThreads = 1;
	xcfg.memoryMode = o->memory_mode;
	xcfg.pciMmioBridge = o->pci_mmio_bridge != 0;
	xcfg.endpointConfig = &ncfg;

	bool use_pci_mmio = o->pci_mmio_bridge != 0;
	xio::nvme_ep::nvmeDoorbellParams dbp;
	dbp.doorbellOffset =
		xio::nvme_ep::doorbellBase +
		(2 * fdat->nvme_qid * xio::nvme_ep::doorBellStride);
	if (o->nvme_target_bdf)
		dbp.nvmeTargetBdf = (uint16_t)o->nvme_target_bdf;
	else {
		uint16_t bdf = 0;
		if (xio::detectBdfFromDevice(f->file_name, &bdf) != 0)
			return hipErrorInvalidValue;
		dbp.nvmeTargetBdf = bdf;
	}
	dbp.shadowBufferVirt =
		use_pci_mmio ? fdat->mmio_shadow_gpu : nullptr;
	dbp.nvmeBar0Gpu = use_pci_mmio ? nullptr : fdat->nvme_bar0_gpu;
	dbp.usePciMmioBridge = use_pci_mmio;

	h = rocm_xio_run_host_burst(xcfg, &ncfg, bp, iop, dbp);

	if (prp_pool)
		(void)hipHostFree(prp_pool);

	if (h != hipSuccess)
		return h;

	if (ddir == DDIR_READ) {
		h = hipMemcpy(priv->gpu_buf,
			      priv->buf_info.gpuPtr ? priv->buf_info.gpuPtr
						    : priv->buf_info.hostPtr,
			      (size_t)buflen, hipMemcpyDefault);
		if (h != hipSuccess)
			return h;
		h = hipMemcpy(host_ptr, priv->gpu_buf, (size_t)buflen,
			      hipMemcpyDeviceToHost);
	}

	return h;
}

} /* namespace */

static enum fio_q_status rocm_xio_queue(struct thread_data *td,
					struct io_u *io_u)
{
	hipError_t h;

	fio_ro_check(td, io_u);

	switch (io_u->ddir) {
	case DDIR_SYNC:
		if (fsync(io_u->file->fd) != 0) {
			io_u->error = errno;
			log_err("fsync: err=%d\n", errno);
		}
		break;

	case DDIR_DATASYNC:
		if (fdatasync(io_u->file->fd) != 0) {
			io_u->error = errno;
			log_err("fdatasync: err=%d\n", errno);
		}
		break;

	case DDIR_READ:
	case DDIR_WRITE:
		h = rocm_xio_submit_one(td, io_u->file, io_u->ddir, io_u->offset,
					io_u->xfer_buf, io_u->xfer_buflen);
		if (h != hipSuccess)
			io_u->error = EIO;
		break;

	default:
		io_u->error = EINVAL;
		break;
	}

	if (io_u->error != 0)
		td_verror(td, io_u->error, "xfer");

	return FIO_Q_COMPLETED;
}

static struct fio_option options[] = {
	{
		.name = "gpu_dev_ids",
		.lname = "rocm-xio engine gpu dev ids",
		.type = FIO_OPT_STR_STORE,
		.off1 = offsetof(struct rocm_xio_options, gpu_ids),
		.help = "HIP device IDs, one per subjob, separated by " GPU_ID_SEP,
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_memory_mode",
		.lname = "rocm-xio memory mode bitmask",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, memory_mode),
		.help = "rocm-xio XIO_MEM_MODE_* bitmask (see ROCm rocm-xio docs)",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_nsid",
		.lname = "rocm-xio NVMe namespace id",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, nsid),
		.help = "Target NVMe namespace ID",
		.def = "1",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_queue_id",
		.lname = "rocm-xio NVMe I/O queue id",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, queue_id),
		.help = "NVMe queue id (0 = use last available I/O queue)",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_queue_length",
		.lname = "rocm-xio NVMe queue depth",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, queue_length),
		.help = "SQ/CQ depth (power of two)",
		.def = "64",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_batch_size",
		.lname = "rocm-xio SQ batch doorbell",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, batch_size),
		.help = "SQEs per doorbell (1 recommended for this engine)",
		.def = "1",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_pci_mmio_bridge",
		.lname = "rocm-xio PCI MMIO bridge doorbells",
		.type = FIO_OPT_BOOL,
		.off1 = offsetof(struct rocm_xio_options, pci_mmio_bridge),
		.help = "Use PCI MMIO bridge for doorbells (VM / QEMU)",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_mmio_bridge_bdf",
		.lname = "rocm-xio MMIO bridge BDF",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, mmio_bridge_bdf),
		.help = "PCI MMIO bridge BDF 0xBBDD; 0 = default / auto",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_nvme_target_bdf",
		.lname = "rocm-xio NVMe BDF override",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, nvme_target_bdf),
		.help = "NVMe BDF 0xBBDD; 0 = detect from filename",
		.def = "0",
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
	.init = rocm_xio_init,
	.queue = rocm_xio_queue,
	.open_file = rocm_xio_open_file,
	.close_file = rocm_xio_close_file,
	.iomem_alloc = rocm_xio_iomem_alloc,
	.iomem_free = rocm_xio_iomem_free,
	.flags = FIO_SYNCIO | FIO_RAWIO | FIO_MEMALIGN,
	.options = options,
	.option_struct_size = sizeof(struct rocm_xio_options),
};

static void fio_init fio_rocm_xio_register(void)
{
	register_ioengine(&ioengine);
}

static void fio_exit fio_rocm_xio_unregister(void)
{
	unregister_ioengine(&ioengine);
}
