/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "rocm_xio_shim.h"

#include <cerrno>
#include <climits>
#include <cstring>
#include <memory>
#include <string>

#include <hip/hip_runtime.h>

#if defined(__has_include)
#if __has_include(<rocm-xio/xio.h>)
#include <rocm-xio/endpoints/nvme-ep/nvme-ep.h>
#include <rocm-xio/xio.h>
#else
#include <nvme-ep.h>
#include <xio.h>
#endif
#else
#include <nvme-ep.h>
#include <xio.h>
#endif

struct fio_rocm_xio_ctx {
	struct fio_rocm_xio_options opts;
	std::unique_ptr<xio::XioEndpoint> endpoint;
	xio::nvme_ep::nvmeEpConfig *nvme_cfg;
	xio::XioEndpointConfig base_cfg;
	std::string last_error;
	unsigned int lba_size;
};

static int set_error(struct fio_rocm_xio_ctx *ctx, const std::string &err)
{
	if (ctx)
		ctx->last_error = err;

	return -1;
}

int fio_rocm_xio_init_ctx(const struct fio_rocm_xio_options *opts,
			  struct fio_rocm_xio_ctx **out_ctx)
{
	struct fio_rocm_xio_ctx *ctx = NULL;
	std::string err;

	if (!opts || !opts->controller || !out_ctx)
		return -1;

	ctx = new fio_rocm_xio_ctx();
	ctx->opts = *opts;
	ctx->nvme_cfg = NULL;
	ctx->lba_size = 0;
	*out_ctx = NULL;

	if (ctx->opts.gpu_id >= 0) {
		hipError_t hret = hipSetDevice(ctx->opts.gpu_id);

		if (hret != hipSuccess) {
			err = std::string("hipSetDevice failed: ") +
			      hipGetErrorString(hret);
			delete ctx;
			return -1;
		}
	}

	ctx->endpoint = xio::createEndpoint("nvme-ep");
	if (!ctx->endpoint) {
		delete ctx;
		return -1;
	}

	ctx->nvme_cfg = static_cast<xio::nvme_ep::nvmeEpConfig *>(
		ctx->endpoint->initializeEndpointConfig());
	if (!ctx->nvme_cfg) {
		delete ctx;
		return -1;
	}

	ctx->nvme_cfg->controller = ctx->opts.controller;
	ctx->nvme_cfg->queueId = ctx->opts.queue_id;
	ctx->nvme_cfg->queueLength = ctx->opts.queue_length;
	ctx->nvme_cfg->ioParams.nsid = ctx->opts.nsid;
	ctx->nvme_cfg->ioParams.lfsrSeed = ctx->opts.lfsr_seed;
	ctx->nvme_cfg->ioParams.infiniteMode = false;
	ctx->nvme_cfg->ioParams.accessPattern = "sequential";
	ctx->nvme_cfg->ioParams.readIo = 1;
	ctx->nvme_cfg->ioParams.writeIo = 0;
	ctx->nvme_cfg->bufferParams.bufferSize = 4096;
	ctx->nvme_cfg->doorbellParams.usePciMmioBridge =
		ctx->opts.use_pci_mmio_bridge;

	std::memset(&ctx->base_cfg, 0, sizeof(ctx->base_cfg));
	ctx->base_cfg.numThreads = 1;
	ctx->base_cfg.memoryMode = ctx->opts.memory_mode;
	ctx->base_cfg.pciMmioBridge = ctx->opts.use_pci_mmio_bridge;
	ctx->base_cfg.verbose = ctx->opts.verbose;
	ctx->base_cfg.endpointConfig = ctx->nvme_cfg;

	ctx->endpoint->applyCommonConfig(ctx->nvme_cfg, &ctx->base_cfg);
	err = ctx->endpoint->validateConfig(ctx->nvme_cfg);
	if (!err.empty()) {
		set_error(ctx, err);
		delete ctx;
		return -1;
	}

	ctx->lba_size = ctx->nvme_cfg->ioParams.lbaSize;
	if (!ctx->lba_size) {
		set_error(ctx, "rocm-xio reported invalid lba size");
		delete ctx;
		return -1;
	}

	*out_ctx = ctx;
	return 0;
}

void fio_rocm_xio_destroy_ctx(struct fio_rocm_xio_ctx *ctx)
{
	if (!ctx)
		return;

	delete ctx;
}

int fio_rocm_xio_submit(struct fio_rocm_xio_ctx *ctx, int is_write,
			uint64_t offset, size_t len)
{
	hipError_t hret;
	uint64_t lbas64;

	if (!ctx || !ctx->nvme_cfg || !ctx->lba_size || !len)
		return set_error(ctx, "invalid rocm-xio submit arguments");

	if ((offset % ctx->lba_size) || (len % ctx->lba_size))
		return set_error(ctx, "offset/length must be lba aligned");

	lbas64 = len / ctx->lba_size;
	if (!lbas64)
		return set_error(ctx, "zero-lba transfer is not valid");
	if (lbas64 > UINT_MAX)
		return set_error(ctx, "transfer exceeds rocm-xio lba limits");
	ctx->nvme_cfg->ioParams.baseLba = offset / ctx->lba_size;
	ctx->nvme_cfg->ioParams.lbasPerIo = lbas64;
	ctx->nvme_cfg->ioParams.readIo = is_write ? 0 : 1;
	ctx->nvme_cfg->ioParams.writeIo = is_write ? 1 : 0;
	ctx->nvme_cfg->bufferParams.bufferSize = len;
	ctx->base_cfg.endpointConfig = ctx->nvme_cfg;
	ctx->base_cfg.iterations = ctx->endpoint->getIterations(ctx->nvme_cfg);

	hret = ctx->endpoint->run(&ctx->base_cfg);
	if (hret != hipSuccess) {
		ctx->last_error = std::string("rocm-xio run failed: ") +
				  hipGetErrorString(hret);
		return -1;
	}

	return 0;
}

int fio_rocm_xio_get_lba_size(const struct fio_rocm_xio_ctx *ctx,
			      unsigned int *lba_size)
{
	if (!ctx || !lba_size || !ctx->lba_size)
		return -1;

	*lba_size = ctx->lba_size;
	return 0;
}

const char *fio_rocm_xio_last_error(const struct fio_rocm_xio_ctx *ctx)
{
	if (!ctx || ctx->last_error.empty())
		return "rocm-xio unknown error";

	return ctx->last_error.c_str();
}
