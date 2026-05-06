/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "rocm_xio_shim.h"

#include <cerrno>
#include <climits>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

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

struct fio_rocm_xio_session {
	struct fio_rocm_xio_session_opts opts;
	xio::nvme_ep::nvmePersistentSession *session;
	std::string last_error;
	unsigned int lba_size;
	uint64_t capacity_lbas;
	std::deque<fio_rocm_xio_completion> completions;
};

static int set_error(struct fio_rocm_xio_session *ctx, const std::string &err)
{
	if (ctx)
		ctx->last_error = err;

	return -1;
}

int fio_rocm_xio_open_session(const struct fio_rocm_xio_session_opts *opts,
			      struct fio_rocm_xio_session **out_ctx)
{
	struct fio_rocm_xio_session *ctx = NULL;
	std::string err;

	if (!opts || !opts->controller || !out_ctx)
		return -1;

	ctx = new fio_rocm_xio_session();
	ctx->opts = *opts;
	ctx->session = NULL;
	ctx->lba_size = 0;
	ctx->capacity_lbas = 0;
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

	xio::nvme_ep::nvmePersistentOptions popts = {};
	popts.controller = ctx->opts.controller;
	popts.queueId = ctx->opts.queue_id;
	popts.queueLength = ctx->opts.queue_length;
	popts.nsid = ctx->opts.nsid;
	popts.memoryMode = ctx->opts.memory_mode;
	popts.lfsrSeed = ctx->opts.lfsr_seed;
	popts.maxTransferBytes = 1024 * 1024;
	popts.ringDepth = ctx->opts.ring_depth;
	popts.batchSize = ctx->opts.batch_size;
	popts.sqBatchSize = ctx->opts.sq_batch_size;
	popts.cqBatchSize = ctx->opts.cq_batch_size;
	popts.precomputePrps = ctx->opts.precompute_prps;
	popts.gpuId = ctx->opts.gpu_id;
	popts.usePciMmioBridge = ctx->opts.use_pci_mmio_bridge;
	popts.verbose = ctx->opts.verbose;
	if (xio::nvme_ep::openPersistentSession(&popts, &ctx->session) < 0) {
		set_error(ctx, "failed to open rocm-xio persistent session");
		delete ctx;
		return -1;
	}

	xio::nvme_ep::nvmePersistentInfo pinfo = {};
	if (xio::nvme_ep::getPersistentInfo(ctx->session, &pinfo) < 0) {
		set_error(ctx, "failed to query rocm-xio persistent session");
		xio::nvme_ep::closePersistentSession(ctx->session);
		delete ctx;
		return -1;
	}
	ctx->lba_size = pinfo.lbaSize;
	ctx->capacity_lbas = pinfo.capacityLbas;
	if (!ctx->lba_size) {
		set_error(ctx, "rocm-xio reported invalid lba size");
		xio::nvme_ep::closePersistentSession(ctx->session);
		delete ctx;
		return -1;
	}

	*out_ctx = ctx;
	return 0;
}

void fio_rocm_xio_close_session(struct fio_rocm_xio_session *ctx)
{
	if (!ctx)
		return;

	xio::nvme_ep::closePersistentSession(ctx->session);
	delete ctx;
}

int fio_rocm_xio_get_namespace_info(struct fio_rocm_xio_session *ctx,
				    struct fio_rocm_xio_namespace_info *info)
{
	if (!ctx || !info || !ctx->lba_size)
		return -1;

	info->lba_size = ctx->lba_size;
	info->capacity_lbas = ctx->capacity_lbas;
	info->capacity_bytes = ctx->capacity_lbas * ctx->lba_size;
	return 0;
}

int fio_rocm_xio_get_phase_stats(struct fio_rocm_xio_session *ctx,
				 struct fio_rocm_xio_phase_stats *stats)
{
	xio::nvme_ep::nvmePersistentPhaseStats pstats = {};

	if (!ctx || !stats || !ctx->session)
		return -1;

	if (xio::nvme_ep::getPersistentPhaseStats(ctx->session, &pstats) < 0)
		return -1;

	stats->idle_wait = pstats.idleWait;
	stats->desc_load = pstats.descLoad;
	stats->prp_build = pstats.prpBuild;
	stats->sqe_build = pstats.sqeBuild;
	stats->sqe_write = pstats.sqeWrite;
	stats->sq_fence = pstats.sqFence;
	stats->sq_doorbell = pstats.sqDoorbell;
	stats->cq_poll = pstats.cqPoll;
	stats->verify = pstats.verify;
	stats->cq_doorbell = pstats.cqDoorbell;
	stats->completion_publish = pstats.completionPublish;
	stats->io_count = pstats.ioCount;
	stats->batch_count = pstats.batchCount;
	stats->submitted_count = pstats.submittedCount;
	stats->completed_count = pstats.completedCount;
	stats->poll_iterations = pstats.pollIterations;
	stats->timeout_count = pstats.timeoutCount;
	stats->error_count = pstats.errorCount;
	stats->max_batch = pstats.maxBatch;
	stats->max_polls = pstats.maxPolls;
	stats->gpu_clock_khz = pstats.gpuClockKHz;
	return 0;
}

int fio_rocm_xio_reset_phase_stats(struct fio_rocm_xio_session *ctx)
{
	if (!ctx || !ctx->session)
		return -1;

	return xio::nvme_ep::resetPersistentPhaseStats(ctx->session) < 0 ? -1 : 0;
}

int fio_rocm_xio_submit_desc(struct fio_rocm_xio_session *ctx,
			     const struct fio_rocm_xio_io_desc *desc)
{
	uint64_t lbas64;
	struct fio_rocm_xio_completion comp = { 0 };

	if (!ctx || !desc || !ctx->session || !ctx->lba_size || !desc->len)
		return set_error(ctx, "invalid rocm-xio submit arguments");

	comp.user_data = desc->user_data;
	comp.bytes = desc->len;

	if ((desc->offset % ctx->lba_size) || (desc->len % ctx->lba_size)) {
		comp.error = EINVAL;
		ctx->completions.push_back(comp);
		return set_error(ctx, "offset/length must be lba aligned");
	}

	lbas64 = desc->len / ctx->lba_size;
	if (!lbas64) {
		comp.error = EINVAL;
		ctx->completions.push_back(comp);
		return set_error(ctx, "zero-lba transfer is not valid");
	}
	if (lbas64 > UINT_MAX) {
		comp.error = EINVAL;
		ctx->completions.push_back(comp);
		return set_error(ctx, "transfer exceeds rocm-xio lba limits");
	}

	xio::nvme_ep::nvmePersistentIo pio = {};
	xio::nvme_ep::nvmePersistentCompletion pcomp = {};
	pio.userData = desc->user_data;
	pio.offset = desc->offset;
	pio.len = desc->len;
	pio.isWrite = desc->is_write;
	pio.verify = desc->do_verify;

	if (xio::nvme_ep::submitPersistent(ctx->session, &pio, &pcomp) < 0) {
		ctx->last_error = "rocm-xio persistent submit failed";
		comp.error = EIO;
		ctx->completions.push_back(comp);
		return -1;
	}

	comp.error = pcomp.error;
	comp.gpu_elapsed_ns = pcomp.gpuElapsedNs;
	comp.nvme_status = pcomp.nvmeStatus;
	comp.verify_pass = pcomp.verifyPass;
	comp.verify_fail = pcomp.verifyFail;
	ctx->completions.push_back(comp);
	return 0;
}

int fio_rocm_xio_post_desc(struct fio_rocm_xio_session *ctx,
			   const struct fio_rocm_xio_io_desc *desc)
{
	uint64_t lbas64;
	xio::nvme_ep::nvmePersistentIo pio = {};

	if (!ctx || !desc || !ctx->session || !ctx->lba_size || !desc->len)
		return set_error(ctx, "invalid rocm-xio post arguments");

	if ((desc->offset % ctx->lba_size) || (desc->len % ctx->lba_size))
		return set_error(ctx, "offset/length must be lba aligned");

	lbas64 = desc->len / ctx->lba_size;
	if (!lbas64)
		return set_error(ctx, "zero-lba transfer is not valid");
	if (lbas64 > UINT_MAX)
		return set_error(ctx, "transfer exceeds rocm-xio lba limits");

	pio.userData = desc->user_data;
	pio.offset = desc->offset;
	pio.len = desc->len;
	pio.isWrite = desc->is_write;
	pio.verify = desc->do_verify;

	return xio::nvme_ep::postPersistent(ctx->session, &pio);
}

int fio_rocm_xio_reap(struct fio_rocm_xio_session *ctx, unsigned int min,
		      unsigned int max, struct fio_rocm_xio_completion *comps,
		      const struct timespec *)
{
	unsigned int done = 0;
	std::vector<xio::nvme_ep::nvmePersistentCompletion> pcomps;

	if (!ctx || !comps || !max)
		return -1;

	pcomps.resize(max);
	int ret = xio::nvme_ep::reapPersistent(ctx->session, min, max,
					       pcomps.data());
	if (ret < 0)
		return ret;
	for (int i = 0; i < ret; i++) {
		comps[done].user_data = pcomps[i].userData;
		comps[done].error = pcomps[i].error;
		comps[done].bytes = pcomps[i].bytes;
		comps[done].gpu_elapsed_ns = pcomps[i].gpuElapsedNs;
		comps[done].nvme_status = pcomps[i].nvmeStatus;
		comps[done].verify_pass = pcomps[i].verifyPass;
		comps[done].verify_fail = pcomps[i].verifyFail;
		done++;
	}

	while (done < max && !ctx->completions.empty()) {
		comps[done++] = ctx->completions.front();
		ctx->completions.pop_front();
	}

	return done >= min ? (int) done : (int) done;
}

int fio_rocm_xio_copy_write_payload(struct fio_rocm_xio_session *,
				    const struct fio_rocm_xio_io_desc *,
				    const void *)
{
	return -ENOTSUP;
}

int fio_rocm_xio_copy_read_payload(struct fio_rocm_xio_session *,
				   const struct fio_rocm_xio_io_desc *, void *)
{
	return -ENOTSUP;
}

int fio_rocm_xio_submit(struct fio_rocm_xio_session *ctx, int is_write,
			uint64_t offset, size_t len)
{
	struct fio_rocm_xio_io_desc desc = { 0 };

	desc.offset = offset;
	desc.len = len;
	desc.is_write = is_write;
	if (fio_rocm_xio_submit_desc(ctx, &desc) < 0)
		return -1;
	if (!ctx->completions.empty())
		ctx->completions.pop_front();
	return 0;
}

int fio_rocm_xio_get_lba_size(const struct fio_rocm_xio_session *ctx,
			      unsigned int *lba_size)
{
	if (!ctx || !lba_size || !ctx->lba_size)
		return -1;

	*lba_size = ctx->lba_size;
	return 0;
}

const char *fio_rocm_xio_last_error(const struct fio_rocm_xio_session *ctx)
{
	if (!ctx || ctx->last_error.empty())
		return "rocm-xio unknown error";

	return ctx->last_error.c_str();
}

int fio_rocm_xio_init_ctx(const struct fio_rocm_xio_session_opts *opts,
			  struct fio_rocm_xio_session **out_ctx)
{
	return fio_rocm_xio_open_session(opts, out_ctx);
}

void fio_rocm_xio_destroy_ctx(struct fio_rocm_xio_session *ctx)
{
	fio_rocm_xio_close_session(ctx);
}
