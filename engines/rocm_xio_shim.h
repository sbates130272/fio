/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#ifndef FIO_ROCM_XIO_SHIM_H
#define FIO_ROCM_XIO_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum fio_rocm_xio_verify_mode {
	FIO_ROCM_XIO_VERIFY_NONE = 0,
	FIO_ROCM_XIO_VERIFY_LFSR = 1,
	FIO_ROCM_XIO_VERIFY_FIO_BUFFER = 2,
};

struct fio_rocm_xio_session_opts {
	const char *controller;
	const char *filename;
	unsigned int queue_id;
	unsigned int queue_length;
	unsigned int nsid;
	unsigned int lfsr_seed;
	unsigned int batch_size;
	unsigned int sq_batch_size;
	unsigned int cq_batch_size;
	unsigned int memory_mode;
	unsigned int verify_mode;
	int precompute_prps;
	unsigned int ring_depth;
	int gpu_id;
	int use_pci_mmio_bridge;
	int verbose;
	int profile;
};

struct fio_rocm_xio_io_desc {
	uint64_t user_data;
	uint64_t offset;
	uint32_t len;
	uint8_t is_write;
	uint8_t do_verify;
};

struct fio_rocm_xio_completion {
	uint64_t user_data;
	int error;
	uint64_t bytes;
	uint64_t gpu_elapsed_ns;
	uint16_t nvme_status;
	uint64_t verify_pass;
	uint64_t verify_fail;
};

struct fio_rocm_xio_namespace_info {
	uint64_t capacity_bytes;
	uint64_t capacity_lbas;
	unsigned int lba_size;
};

struct fio_rocm_xio_phase_stats {
	uint64_t idle_wait;
	uint64_t desc_load;
	uint64_t prp_build;
	uint64_t sqe_build;
	uint64_t sqe_write;
	uint64_t sq_fence;
	uint64_t sq_doorbell;
	uint64_t cq_poll;
	uint64_t verify;
	uint64_t cq_doorbell;
	uint64_t completion_publish;
	uint64_t io_count;
	uint64_t batch_count;
	uint64_t submitted_count;
	uint64_t completed_count;
	uint64_t poll_iterations;
	uint64_t timeout_count;
	uint64_t error_count;
	uint64_t max_batch;
	uint64_t max_polls;
	uint32_t gpu_clock_khz;
};

struct fio_rocm_xio_session;

int fio_rocm_xio_open_session(const struct fio_rocm_xio_session_opts *opts,
			      struct fio_rocm_xio_session **out_session);
void fio_rocm_xio_close_session(struct fio_rocm_xio_session *session);
int fio_rocm_xio_get_namespace_info(struct fio_rocm_xio_session *session,
				    struct fio_rocm_xio_namespace_info *info);
int fio_rocm_xio_get_phase_stats(struct fio_rocm_xio_session *session,
				 struct fio_rocm_xio_phase_stats *stats);
int fio_rocm_xio_reset_phase_stats(struct fio_rocm_xio_session *session);
int fio_rocm_xio_submit_desc(struct fio_rocm_xio_session *session,
			     const struct fio_rocm_xio_io_desc *desc);
int fio_rocm_xio_post_desc(struct fio_rocm_xio_session *session,
			   const struct fio_rocm_xio_io_desc *desc);
int fio_rocm_xio_reap(struct fio_rocm_xio_session *session, unsigned int min,
		      unsigned int max, struct fio_rocm_xio_completion *comps,
		      const struct timespec *timeout);
int fio_rocm_xio_copy_write_payload(struct fio_rocm_xio_session *session,
				    const struct fio_rocm_xio_io_desc *desc,
				    const void *src);
int fio_rocm_xio_copy_read_payload(struct fio_rocm_xio_session *session,
				   const struct fio_rocm_xio_io_desc *desc,
				   void *dst);
const char *fio_rocm_xio_last_error(const struct fio_rocm_xio_session *session);

typedef struct fio_rocm_xio_session fio_rocm_xio_ctx;
typedef struct fio_rocm_xio_session_opts fio_rocm_xio_options;

int fio_rocm_xio_init_ctx(const struct fio_rocm_xio_session_opts *opts,
			  struct fio_rocm_xio_session **out_ctx);
void fio_rocm_xio_destroy_ctx(struct fio_rocm_xio_session *ctx);
int fio_rocm_xio_submit(struct fio_rocm_xio_session *ctx, int is_write,
			uint64_t offset, size_t len);
int fio_rocm_xio_get_lba_size(const struct fio_rocm_xio_session *ctx,
			      unsigned int *lba_size);

#ifdef __cplusplus
}
#endif

#endif
