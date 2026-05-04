/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef FIO_ROCM_XIO_BRIDGE_H
#define FIO_ROCM_XIO_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

struct rocm_xio_submit_opts {
	const char *controller;
	uint64_t base_lba;
	uint32_t lbas_per_io;
	uint32_t nsid;
	uint16_t queue_id;
	uint16_t queue_length;
	uint16_t num_queues;
	uint32_t memory_mode;
	uint32_t batch_size;
	uint32_t lfsr_seed;
	uint32_t use_pci_mmio_bridge;
	uint32_t verify;
	uint32_t random_access;
	uint32_t do_read;
	uint32_t do_write;
};

#ifdef __cplusplus
extern "C" {
#endif

int rocm_xio_query_lba_size(const char *controller, uint32_t nsid,
			    uint32_t *lba_size, char *errbuf,
			    size_t errbuf_len);

int rocm_xio_query_namespace_capacity(const char *controller, uint32_t nsid,
				      uint64_t *capacity_lbas, char *errbuf,
				      size_t errbuf_len);

int rocm_xio_submit(const struct rocm_xio_submit_opts *opts,
		    uint32_t *verify_pass, uint32_t *verify_fail,
		    char *errbuf, size_t errbuf_len);

#ifdef __cplusplus
}
#endif

#endif
