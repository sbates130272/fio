/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef FIO_ROCM_XIO_BRIDGE_H
#define FIO_ROCM_XIO_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct rocm_xio_cfg {
	const char *controller;
	unsigned int queue_id;
	unsigned int queue_len;
	unsigned int nsid;
	int gpu_id;
	unsigned int allow_rootfs;
	unsigned int memory_mode;
	unsigned int use_pci_mmio_bridge;
	unsigned int lba_size;
	unsigned int lfsr_seed;
	unsigned int subjob_number;
	size_t read_buf_size;
	size_t write_buf_size;
};

struct rocm_xio_io {
	void *buffer;
	size_t len;
	uint64_t offset;
	int is_read;
};

struct rocm_xio_handle;

int rocm_xio_handle_create(const struct rocm_xio_cfg *cfg,
			   struct rocm_xio_handle **handle,
			   char *err, size_t errlen);

void rocm_xio_handle_destroy(struct rocm_xio_handle *handle);

int rocm_xio_handle_submit(struct rocm_xio_handle *handle,
			   const struct rocm_xio_io *io,
			   char *err, size_t errlen);

int rocm_xio_query_namespace(const char *controller, uint32_t nsid,
			     uint32_t *lba_size, uint64_t *capacity_lbas,
			     char *err, size_t errlen);

uint32_t rocm_xio_handle_lba_size(const struct rocm_xio_handle *handle);
uint64_t rocm_xio_handle_capacity_lbas(const struct rocm_xio_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
