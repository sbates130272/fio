/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "rocm_xio_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <new>
#include <vector>

#include <hip/hip_runtime.h>

#include <xio.h>
#include <endpoints/nvme-ep/nvme-ep.h>

struct rocm_xio_handle {
	std::string controller;
	uint16_t queue_id;
	uint16_t queue_len;
	uint32_t nsid;
	uint32_t lba_size;
	uint64_t capacity_lbas;
	uint16_t nvme_bdf;
	xio::nvme_ep::nvme_queue_info queue_info;
	xio::nvme_ep::nvmeDoorbellParams doorbell_params;
	volatile xio::nvme_ep::sqeType *sqes;
	volatile xio::nvme_ep::cqeType *cqes;
	uint16_t sq_tail;
	uint16_t cq_head;
	uint8_t cq_phase;
	uint16_t cmd_id;
	size_t io_buf_size;
	xio::xioBufferInfo io_buf;
	xio::nvme_ep::nvmeEpConfig cleanup_config;
};

using rocm_xio_handle = struct rocm_xio_handle;

static int rocm_xio_errno_or_fallback(int code, int fallback)
{
	if (code > 0)
		return -code;
	if (!code)
		return fallback;
	return code;
}

static void rocm_xio_set_error(char *err, size_t errlen, const char *fmt, ...)
{
	va_list ap;

	if (!err || !errlen)
		return;

	va_start(ap, fmt);
	vsnprintf(err, errlen, fmt, ap);
	va_end(ap);
}

static int rocm_xio_query_namespace_impl(const char *controller, uint32_t nsid,
					 uint32_t *lba_size, uint64_t *capacity_lbas,
					 char *err, size_t errlen)
{
	unsigned int lba = 0;
	uint64_t capacity = 0;
	int rc;

	rc = xio::nvme_ep::queryLbaSize(controller, nsid, &lba);
	if (rc) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio queryLbaSize(%s, nsid=%u) failed: %s",
				   controller, nsid, strerror(-rc));
		return rc;
	}

	rc = xio::nvme_ep::queryNamespaceCapacity(controller, nsid, &capacity);
	if (rc) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio queryNamespaceCapacity(%s, nsid=%u) failed: %s",
				   controller, nsid, strerror(-rc));
		return rc;
	}

	if (!lba || !capacity) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio returned invalid namespace geometry (lba=%u, cap=%llu)",
				   lba, (unsigned long long) capacity);
		return -EINVAL;
	}

	if (lba_size)
		*lba_size = lba;
	if (capacity_lbas)
		*capacity_lbas = capacity;

	return 0;
}

int rocm_xio_query_namespace(const char *controller, uint32_t nsid,
			     uint32_t *lba_size, uint64_t *capacity_lbas,
			     char *err, size_t errlen)
{
	if (!controller || !*controller) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio controller path is required");
		return -EINVAL;
	}
	if (!nsid) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio nsid must be > 0");
		return -EINVAL;
	}

	return rocm_xio_query_namespace_impl(controller, nsid, lba_size,
					     capacity_lbas, err, errlen);
}

int rocm_xio_handle_create(const struct rocm_xio_cfg *cfg,
			   rocm_xio_handle **handle,
			   char *err, size_t errlen)
{
	rocm_xio_handle *h = NULL;
	void *bar_cpu = nullptr;
	void *bar_gpu = nullptr;
	void *shadow_cpu = nullptr;
	void *shadow_gpu = nullptr;
	uint16_t queue_id;
	uint16_t queue_len;
	uint32_t nsid;
	uint16_t nvme_bdf = 0;
	unsigned int data_memory_mode;
	size_t max_buf_size;
	int rc = 0;

	if (!cfg || !handle) {
		rocm_xio_set_error(err, errlen, "rocm-xio config is required");
		return -EINVAL;
	}
	if (!cfg->controller || !*cfg->controller) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio controller path is required");
		return -EINVAL;
	}

	queue_len = cfg->queue_len ? cfg->queue_len : 64;
	if (!queue_len || (queue_len & (queue_len - 1))) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio queue length must be a power of two");
		return -EINVAL;
	}

	nsid = cfg->nsid ? cfg->nsid : 1;
	queue_id = cfg->queue_id;

	if (cfg->gpu_id >= 0) {
		hipError_t herr = hipSetDevice(cfg->gpu_id);

		if (herr != hipSuccess) {
			rocm_xio_set_error(err, errlen,
					   "rocm-xio hipSetDevice(%d) failed: %s",
					   cfg->gpu_id, hipGetErrorString(herr));
			return -EINVAL;
		}
	}

	rc = xio::detectBdfFromDevice(cfg->controller, &nvme_bdf);
	if (rc) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio detectBdfFromDevice(%s) failed: %s",
				   cfg->controller, strerror(-rc));
		return rc;
	}

	if (!cfg->allow_rootfs) {
		rc = xio::nvme_ep::checkRootfs(cfg->controller);
		if (rc) {
			rocm_xio_set_error(err, errlen,
					   "rocm-xio refuses to run on root filesystem device %s",
					   cfg->controller);
			return -EPERM;
		}
	}

	if (!queue_id) {
		rc = xio::nvme_ep::queryMaxQueueId(cfg->controller, &queue_id);
		if (rc) {
			int qid_rc = rocm_xio_errno_or_fallback(rc, -EIO);
			rocm_xio_set_error(err, errlen,
					   "rocm-xio queryMaxQueueId(%s) failed: %s",
					   cfg->controller, strerror(-qid_rc));
			return qid_rc;
		}
	}

	h = new (std::nothrow) rocm_xio_handle();
	if (!h) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio failed to allocate handle");
		return -ENOMEM;
	}
	h->controller = cfg->controller;
	h->queue_id = queue_id;
	h->queue_len = queue_len;
	h->nsid = nsid;
	h->nvme_bdf = nvme_bdf;
	h->sq_tail = 0;
	h->cq_head = 0;
	h->cq_phase = 1;
	h->cmd_id = 1;
	h->io_buf_size = 0;
	memset(&h->queue_info, 0, sizeof(h->queue_info));
	memset(&h->io_buf, 0, sizeof(h->io_buf));

	if (cfg->use_pci_mmio_bridge) {
		uint16_t bridge_bdf = 0;

		rc = xio::detectPciMmioBridgeBdf(&bridge_bdf);
		if (rc) {
			int bdf_rc = rocm_xio_errno_or_fallback(rc, -EIO);
			rocm_xio_set_error(err, errlen,
					   "rocm-xio detectPciMmioBridgeBdf failed: %s",
					   strerror(-bdf_rc));
			rc = bdf_rc;
			goto out_err;
		}

		rc = xio::mapMmioBridgeShadowBuffer(bridge_bdf, &shadow_cpu);
		if (rc) {
			int map_rc = rocm_xio_errno_or_fallback(rc, -EIO);
			rocm_xio_set_error(err, errlen,
					   "rocm-xio mapMmioBridgeShadowBuffer failed: %s",
					   strerror(-map_rc));
			rc = map_rc;
			goto out_err;
		}

		rc = xio::registerMmioBridgeShadowBufferForGpu(shadow_cpu, 8192,
							      &shadow_gpu);
		if (rc) {
			int reg_rc = rocm_xio_errno_or_fallback(rc, -EIO);
			rocm_xio_set_error(err, errlen,
					   "rocm-xio registerMmioBridgeShadowBufferForGpu failed: %s",
					   strerror(-reg_rc));
			rc = reg_rc;
			goto out_err;
		}

		h->doorbell_params.usePciMmioBridge = true;
		h->doorbell_params.shadowBufferVirt = shadow_gpu;
		h->doorbell_params.nvmeBar0Gpu = nullptr;
	} else {
		rc = xio::mapPciBar(nvme_bdf, 0, &bar_cpu, &bar_gpu, 8192);
		if (rc) {
			int bar_rc = rocm_xio_errno_or_fallback(rc, -EIO);
			rocm_xio_set_error(err, errlen,
					   "rocm-xio mapPciBar failed: %s",
					   strerror(-bar_rc));
			rc = bar_rc;
			goto out_err;
		}

		h->doorbell_params.usePciMmioBridge = false;
		h->doorbell_params.shadowBufferVirt = nullptr;
		h->doorbell_params.nvmeBar0Gpu = bar_gpu;
	}

	h->doorbell_params.nvmeTargetBdf = nvme_bdf;
	h->doorbell_params.doorbellOffset = xio::nvme_ep::doorbellBase +
					    (2U * queue_id *
					     xio::nvme_ep::doorBellStride);

	rc = xio::nvme_ep::createQueue(cfg->controller, ROCM_XIO_DEVICE_PATH,
				       queue_id, queue_len, nvme_bdf,
				       cfg->memory_mode, &h->queue_info);
	if (rc) {
		int q_rc = rocm_xio_errno_or_fallback(rc, -EIO);
		rocm_xio_set_error(err, errlen,
				   "rocm-xio createQueue failed: %s",
				   strerror(-q_rc));
		rc = q_rc;
		goto out_err;
	}

	h->sqes = static_cast<volatile xio::nvme_ep::sqeType *>(
			h->queue_info.sq_gpu ? h->queue_info.sq_gpu
					     : h->queue_info.sq_virt);
	h->cqes = static_cast<volatile xio::nvme_ep::cqeType *>(
			h->queue_info.cq_gpu ? h->queue_info.cq_gpu
					     : h->queue_info.cq_virt);
	if (!h->sqes || !h->cqes) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio queue pointers are invalid");
		rc = -EINVAL;
		goto out_err;
	}

	rc = rocm_xio_query_namespace_impl(cfg->controller, nsid, &h->lba_size,
					   &h->capacity_lbas, err, errlen);
	if (rc)
		goto out_err;

	max_buf_size = cfg->read_buf_size;
	if (cfg->write_buf_size > max_buf_size)
		max_buf_size = cfg->write_buf_size;
	if (!max_buf_size)
		max_buf_size = h->lba_size;
	data_memory_mode = cfg->memory_mode | XIO_MEM_MODE_DATA_DEVICE;
	rc = xio::allocateGpuAccessibleBuffer(max_buf_size, data_memory_mode,
					      nvme_bdf, cfg->controller,
					      &h->io_buf);
	if (rc != hipSuccess) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio allocateGpuAccessibleBuffer failed: %s",
				   hipGetErrorString((hipError_t) rc));
		rc = -ENOMEM;
		goto out_err;
	}
	if (!h->io_buf.dmaAddr) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio data buffer DMA address is invalid");
		rc = -EIO;
		goto out_err;
	}
	h->io_buf_size = max_buf_size;

	h->cleanup_config.controller = h->controller;
	h->cleanup_config.queueId = queue_id;
	h->cleanup_config.queuesCreated = true;
	h->cleanup_config.queueIds.push_back(queue_id);
	h->cleanup_config.queueInfos.push_back(h->queue_info);

	*handle = h;
	return 0;

out_err:
	if (h)
		rocm_xio_handle_destroy(h);
	return rc ? rc : -EINVAL;
}

void rocm_xio_handle_destroy(rocm_xio_handle *handle)
{
	if (!handle)
		return;

	if (handle->io_buf_size)
		xio::freeGpuAccessibleBuffer(&handle->io_buf);

	if (handle->cleanup_config.queuesCreated)
		(void)xio::nvme_ep::nvme_ep_cleanup_queues(&handle->cleanup_config);

	delete handle;
}

int rocm_xio_handle_submit(rocm_xio_handle *handle,
			   const struct rocm_xio_io *io,
			   char *err, size_t errlen)
{
	uint32_t blocks;
	uint64_t lba;
	uint64_t dma_addr;
	void *gpu_buf;
	uint16_t cid;
	xio::nvme_ep::sqeType sqe = {};
	unsigned poll_count = 0;

	if (!handle || !io || !io->buffer || !io->len) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio invalid I/O request");
		return -EINVAL;
	}
	if (!handle->lba_size) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio unknown LBA size");
		return -EINVAL;
	}
	if (io->offset % handle->lba_size || io->len % handle->lba_size) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio requires offset/length aligned to %u bytes",
				   handle->lba_size);
		return -EINVAL;
	}
	if (io->len > 2 * NVME_PAGE_SIZE) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio currently supports I/O up to %u bytes",
				   2 * NVME_PAGE_SIZE);
		return -E2BIG;
	}
	if (io->len > handle->io_buf_size) {
		rocm_xio_set_error(err, errlen,
				   "rocm-xio transfer length %zu exceeds internal buffer %zu",
				   io->len, handle->io_buf_size);
		return -E2BIG;
	}

	blocks = io->len / handle->lba_size;
	lba = io->offset / handle->lba_size;
	cid = handle->cmd_id++;
	if (!handle->cmd_id)
		handle->cmd_id = 1;

	gpu_buf = handle->io_buf.gpuPtr ? handle->io_buf.gpuPtr : handle->io_buf.hostPtr;
	dma_addr = handle->io_buf.dmaAddr;
	if (io->is_read == 0) {
		if (handle->io_buf.isDeviceMemory) {
			hipError_t herr = hipMemcpy(gpu_buf, io->buffer, io->len,
						    hipMemcpyHostToDevice);
			if (herr != hipSuccess) {
				rocm_xio_set_error(err, errlen,
					   "rocm-xio H2D copy failed: %s",
					   hipGetErrorString(herr));
				return -EIO;
			}
		} else {
			memcpy(gpu_buf, io->buffer, io->len);
		}
	}

	sqe.opcode = io->is_read ? nvme_cmd_read : nvme_cmd_write;
	sqe.command_id = cid;
	sqe.nsid = handle->nsid;
	xio::nvme_ep::calculatePrps(dma_addr, io->len, &sqe);
	xio::nvme_ep::sqeSetup(&sqe, lba, blocks);

	xio::nvme_ep::sqeWrite(sqe, &handle->sqes[handle->sq_tail]);
	handle->sq_tail = (handle->sq_tail + 1) % handle->queue_len;
	xio::nvme_ep::ringDoorbell(handle->sq_tail, handle->doorbell_params);

	while (true) {
		xio::nvme_ep::cqeType cqe;
		uint8_t phase;

		cqe = xio::nvme_ep::cqeRead(&handle->cqes[handle->cq_head]);
		phase = NVME_CQE_STATUS_PHASE(cqe.status);
		if (phase == handle->cq_phase) {
			if (!xio::nvme_ep::cqeOk(&cqe)) {
				rocm_xio_set_error(err, errlen,
						   "rocm-xio completion error: sc=0x%02x sct=0x%02x",
						   xio::nvme_ep::cqeStatusCode(&cqe),
						   xio::nvme_ep::cqeStatusType(&cqe));
				return -EIO;
			}
			break;
		}

		if (++poll_count >= xio::nvme_ep::NVME_EP_MAX_POLLS) {
			rocm_xio_set_error(err, errlen,
					   "rocm-xio completion poll timed out");
			return -ETIMEDOUT;
		}
	}

	handle->cq_head = (handle->cq_head + 1) % handle->queue_len;
	if (!handle->cq_head)
		handle->cq_phase ^= 1;

	xio::nvme_ep::ringDoorbell(handle->cq_head, handle->doorbell_params,
				   handle->doorbell_params.doorbellOffset +
				   xio::nvme_ep::doorBellStride);

	if (io->is_read) {
		if (handle->io_buf.isDeviceMemory) {
			hipError_t herr = hipMemcpy(io->buffer, gpu_buf, io->len,
						    hipMemcpyDeviceToHost);
			if (herr != hipSuccess) {
				rocm_xio_set_error(err, errlen,
					   "rocm-xio D2H copy failed: %s",
					   hipGetErrorString(herr));
				return -EIO;
			}
		} else {
			memcpy(io->buffer, gpu_buf, io->len);
		}
	}

	return 0;
}

uint32_t rocm_xio_handle_lba_size(const rocm_xio_handle *handle)
{
	return handle ? handle->lba_size : 0;
}

uint64_t rocm_xio_handle_capacity_lbas(const rocm_xio_handle *handle)
{
	return handle ? handle->capacity_lbas : 0;
}
