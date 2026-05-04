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

struct fio_rocm_xio_options {
	const char *controller;
	unsigned int queue_id;
	unsigned int queue_length;
	unsigned int nsid;
	unsigned int lfsr_seed;
	unsigned int batch_size;
	unsigned int memory_mode;
	int gpu_id;
	int use_pci_mmio_bridge;
	int verbose;
};

struct fio_rocm_xio_ctx;

int fio_rocm_xio_init_ctx(const struct fio_rocm_xio_options *opts,
			  struct fio_rocm_xio_ctx **out_ctx);
void fio_rocm_xio_destroy_ctx(struct fio_rocm_xio_ctx *ctx);
int fio_rocm_xio_submit(struct fio_rocm_xio_ctx *ctx, int is_write,
			uint64_t offset, size_t len);
int fio_rocm_xio_get_lba_size(const struct fio_rocm_xio_ctx *ctx,
			      unsigned int *lba_size);
const char *fio_rocm_xio_last_error(const struct fio_rocm_xio_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
