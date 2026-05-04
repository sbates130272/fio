/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>

#include "../fio.h"
#include "../optgroup.h"
#include "rocm_xio_bridge.h"

#define ROCM_XIO_ERRSTR_LEN 256

struct rocm_xio_options {
	void *pad;
	char *controller;
	unsigned int queue_id;
	unsigned int queue_len;
	unsigned int nsid;
	int gpu_id;
	unsigned int memory_mode;
	unsigned int use_pci_mmio_bridge;
	unsigned int allow_rootfs;
	unsigned int lba_size;
	unsigned int lfsr_seed;
};

struct rocm_xio_data {
	struct rocm_xio_handle *handle;
	uint32_t lba_size;
	uint64_t capacity_lbas;
	char err[ROCM_XIO_ERRSTR_LEN];
};

static struct fio_option options[] = {
	{
		.name = "rocm_xio_controller",
		.lname = "rocm-xio NVMe controller device path",
		.type = FIO_OPT_STR_STORE,
		.off1 = offsetof(struct rocm_xio_options, controller),
		.help = "NVMe controller device path (for example /dev/nvme0)",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_queue_id",
		.lname = "rocm-xio NVMe IO queue ID",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, queue_id),
		.help = "Queue ID to use (0 means auto-detect max queue ID)",
		.interval = 1,
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_queue_len",
		.lname = "rocm-xio queue length",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, queue_len),
		.help = "Queue depth in entries, must be a power of two",
		.interval = 1,
		.minval = 2,
		.def = "64",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_nsid",
		.lname = "rocm-xio namespace identifier",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, nsid),
		.help = "Namespace ID for IO commands",
		.interval = 1,
		.minval = 1,
		.def = "1",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_gpu_id",
		.lname = "rocm-xio HIP GPU device ID",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, gpu_id),
		.help = "HIP GPU device index used by rocm-xio",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_memory_mode",
		.lname = "rocm-xio memory mode bit mask",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, memory_mode),
		.help = "rocm-xio memory mode bit mask for queue/data placement",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_mmio_bridge",
		.lname = "rocm-xio PCI MMIO bridge doorbell routing",
		.type = FIO_OPT_BOOL,
		.off1 = offsetof(struct rocm_xio_options, use_pci_mmio_bridge),
		.help = "Use PCI MMIO bridge for doorbell writes",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_lba_size",
		.lname = "rocm-xio override namespace LBA size",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, lba_size),
		.help = "Optional override for namespace LBA size in bytes (0 means auto)",
		.interval = 1,
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_allow_rootfs",
		.lname = "allow rocm-xio on root filesystem device",
		.type = FIO_OPT_BOOL,
		.off1 = offsetof(struct rocm_xio_options, allow_rootfs),
		.help = "Allow running against root filesystem NVMe device",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = "rocm_xio_lfsr_seed",
		.lname = "rocm-xio LFSR seed",
		.type = FIO_OPT_INT,
		.off1 = offsetof(struct rocm_xio_options, lfsr_seed),
		.help = "LFSR seed used by rocm-xio path",
		.def = "0",
		.category = FIO_OPT_C_ENGINE,
		.group = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = NULL,
	},
};

static int rocm_xio_setup(struct thread_data *td)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_data *rd;
	uint32_t lba_size = 0;
	uint64_t capacity_lbas = 0;
	int rc;

	if (!o->controller || !*o->controller) {
		log_err("rocm-xio: rocm_xio_controller is required\n");
		return 1;
	}

	rd = calloc(1, sizeof(*rd));
	if (!rd) {
		log_err("rocm-xio: calloc failed: %s\n", strerror(errno));
		return 1;
	}

	rc = rocm_xio_query_namespace(o->controller, o->nsid ? o->nsid : 1,
				      &lba_size, &capacity_lbas,
				      rd->err, sizeof(rd->err));
	if (rc) {
		log_err("rocm-xio: %s\n", rd->err);
		free(rd);
		return 1;
	}

	if (o->lba_size && o->lba_size != lba_size) {
		log_err("rocm-xio: requested LBA size %u does not match namespace LBA %u\n",
			o->lba_size, lba_size);
		free(rd);
		return 1;
	}

	rd->lba_size = lba_size;
	rd->capacity_lbas = capacity_lbas;
	td->io_ops_data = rd;

	return 0;
}

static int rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_data *rd = td->io_ops_data;
	struct rocm_xio_cfg cfg;
	int rc;

	if (!rd)
		return 1;

	memset(&cfg, 0, sizeof(cfg));
	cfg.controller = o->controller;
	cfg.queue_id = o->queue_id;
	cfg.queue_len = o->queue_len;
	cfg.nsid = o->nsid;
	cfg.gpu_id = o->gpu_id;
	cfg.memory_mode = o->memory_mode;
	cfg.use_pci_mmio_bridge = o->use_pci_mmio_bridge;
	cfg.allow_rootfs = o->allow_rootfs;
	cfg.lba_size = rd->lba_size;
	cfg.lfsr_seed = o->lfsr_seed;
	cfg.subjob_number = td->subjob_number;
	cfg.read_buf_size = td_max_bs(td);
	cfg.write_buf_size = td_max_bs(td);

	rc = rocm_xio_handle_create(&cfg, &rd->handle, rd->err, sizeof(rd->err));
	if (rc) {
		log_err("rocm-xio: %s\n", rd->err);
		return 1;
	}

	rd->lba_size = rocm_xio_handle_lba_size(rd->handle);
	rd->capacity_lbas = rocm_xio_handle_capacity_lbas(rd->handle);

	return 0;
}

static int rocm_xio_get_file_size(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_data *rd = td->io_ops_data;

	if (!rd || !rd->lba_size || !rd->capacity_lbas)
		return 1;

	f->real_file_size = rd->capacity_lbas * rd->lba_size;
	fio_file_set_size_known(f);
	f->filetype = FIO_TYPE_BLOCK;
	return 0;
}

static int rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	int rc;

	rc = generic_open_file(td, f);
	if (rc)
		return rc;

	return rocm_xio_get_file_size(td, f);
}

static int rocm_xio_close_file(struct thread_data *td, struct fio_file *f)
{
	return generic_close_file(td, f);
}

static int rocm_xio_get_zoned_model(struct thread_data *td, struct fio_file *f,
				    enum zbd_zoned_model *model)
{
	(void)td;
	(void)f;
	*model = ZBD_NONE;
	return 0;
}

static enum fio_q_status rocm_xio_queue(struct thread_data *td, struct io_u *io_u)
{
	struct rocm_xio_data *rd = td->io_ops_data;
	struct rocm_xio_io io;
	int rc;

	if (!rd || !rd->handle) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "rocm_xio_queue");
		return FIO_Q_COMPLETED;
	}

	fio_ro_check(td, io_u);

	switch (io_u->ddir) {
	case DDIR_READ:
	case DDIR_WRITE:
		if (!io_u->xfer_buf || !io_u->xfer_buflen) {
			io_u->error = EINVAL;
			break;
		}
		io.buffer = io_u->xfer_buf;
		io.len = io_u->xfer_buflen;
		io.offset = io_u->offset;
		io.is_read = (io_u->ddir == DDIR_READ);
		rc = rocm_xio_handle_submit(rd->handle, &io,
					    rd->err, sizeof(rd->err));
		if (rc) {
			io_u->error = -rc;
			if (!io_u->error)
				io_u->error = EIO;
			log_err("rocm-xio: %s\n", rd->err);
		}
		break;
	case DDIR_SYNC:
		rc = fsync(io_u->file->fd);
		if (rc)
			io_u->error = errno;
		break;
	case DDIR_DATASYNC:
		rc = fdatasync(io_u->file->fd);
		if (rc)
			io_u->error = errno;
		break;
	case DDIR_TRIM:
		io_u->error = EOPNOTSUPP;
		break;
	default:
		io_u->error = EINVAL;
		break;
	}

	if (io_u->error)
		td_verror(td, io_u->error, "rocm_xio_queue");

	return FIO_Q_COMPLETED;
}

static void rocm_xio_cleanup(struct thread_data *td)
{
	struct rocm_xio_data *rd = td->io_ops_data;

	if (!rd)
		return;

	if (rd->handle) {
		rocm_xio_handle_destroy(rd->handle);
		rd->handle = NULL;
	}

	free(rd);
	td->io_ops_data = NULL;
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name = "rocm_xio",
	.version = FIO_IOOPS_VERSION,
	.setup = rocm_xio_setup,
	.init = rocm_xio_init,
	.queue = rocm_xio_queue,
	.open_file = rocm_xio_open_file,
	.close_file = rocm_xio_close_file,
	.get_file_size = rocm_xio_get_file_size,
	.get_zoned_model = rocm_xio_get_zoned_model,
	.cleanup = rocm_xio_cleanup,
	.flags = FIO_SYNCIO | FIO_RAWIO | FIO_NODISKUTIL,
	.options = options,
	.option_struct_size = sizeof(struct rocm_xio_options),
};

void fio_init fio_rocm_xio_register(void)
{
	register_ioengine(&ioengine);
}

void fio_exit fio_rocm_xio_unregister(void)
{
	unregister_ioengine(&ioengine);
}
