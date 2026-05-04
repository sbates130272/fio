/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../fio.h"
#include "../optgroup.h"
#include "rocm_xio_shim.h"

struct rocm_xio_options {
	void *pad;
	char *controller;
	unsigned int queue_id;
	unsigned int queue_length;
	unsigned int nsid;
	unsigned int lfsr_seed;
	unsigned int batch_size;
	unsigned int memory_mode;
	int gpu_id;
	int pci_mmio_bridge;
	int verbose;
};

struct rocm_xio_data {
	struct fio_rocm_xio_ctx *ctx;
	unsigned int lba_size;
};

static struct fio_option options[] = {
	{
		.name	= "rocm_xio_controller",
		.lname	= "ROCm XIO NVMe controller",
		.type	= FIO_OPT_STR_STORE,
		.off1	= offsetof(struct rocm_xio_options, controller),
		.help	= "NVMe controller path, e.g. /dev/nvme0",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_gpu_id",
		.lname	= "ROCm XIO GPU ID",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, gpu_id),
		.def	= "-1",
		.help	= "GPU id for submitting xio kernels, -1 means current",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_queue_id",
		.lname	= "ROCm XIO queue id",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, queue_id),
		.def	= "0",
		.help	= "NVMe queue id (0 enables rocm-xio auto-detect)",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_queue_length",
		.lname	= "ROCm XIO queue length",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, queue_length),
		.def	= "64",
		.help	= "NVMe queue length in entries",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_nsid",
		.lname	= "ROCm XIO namespace id",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, nsid),
		.def	= "1",
		.help	= "NVMe namespace id",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_lfsr_seed",
		.lname	= "ROCm XIO LFSR seed",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, lfsr_seed),
		.def	= "0",
		.help	= "LFSR seed used by rocm-xio pattern generator",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_batch_size",
		.lname	= "ROCm XIO batch size",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, batch_size),
		.def	= "1",
		.help	= "SQEs per doorbell ring used by rocm-xio",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_memory_mode",
		.lname	= "ROCm XIO memory mode",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, memory_mode),
		.def	= "0",
		.help	= "rocm-xio memory_mode bitmap",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_pci_mmio_bridge",
		.lname	= "ROCm XIO use pci-mmio-bridge mode",
		.type	= FIO_OPT_BOOL,
		.off1	= offsetof(struct rocm_xio_options, pci_mmio_bridge),
		.def	= "0",
		.help	= "Use rocm-xio PCI MMIO bridge for doorbell writes",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_verbose",
		.lname	= "ROCm XIO verbose mode",
		.type	= FIO_OPT_BOOL,
		.off1	= offsetof(struct rocm_xio_options, verbose),
		.def	= "0",
		.help	= "Enable verbose rocm-xio logging",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = NULL,
	},
};

static int fio_rocm_xio_setup(struct thread_data *td)
{
	struct rocm_xio_data *data;
	struct fio_file *f;

	data = calloc(1, sizeof(*data));
	if (!data)
		return 1;

	td->io_ops_data = data;

	if (!td->files_index) {
		add_file(td, "rocm-xio", 0, 0);
		td->o.nr_files = td->o.nr_files ? : 1;
		td->o.open_files++;
	}

	f = td->files[0];
	f->real_file_size = td->o.size ? td->o.size : (1ULL << 40);
	return 0;
}

static int fio_rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_data *data = td->io_ops_data;
	struct fio_rocm_xio_options opts = { 0 };
	struct fio_file *f;
	int i;

	if (!o->controller) {
		log_err("rocm_xio: rocm_xio_controller must be set\n");
		return 1;
	}

	opts.controller = o->controller;
	opts.queue_id = o->queue_id;
	opts.queue_length = o->queue_length;
	opts.nsid = o->nsid;
	opts.lfsr_seed = o->lfsr_seed;
	opts.batch_size = o->batch_size;
	opts.memory_mode = o->memory_mode;
	opts.gpu_id = o->gpu_id;
	opts.use_pci_mmio_bridge = o->pci_mmio_bridge;
	opts.verbose = o->verbose;

	if (fio_rocm_xio_init_ctx(&opts, &data->ctx) < 0 || !data->ctx) {
		log_err("rocm_xio: failed to initialize rocm-xio context\n");
		return 1;
	}

	if (fio_rocm_xio_get_lba_size(data->ctx, &data->lba_size) < 0) {
		log_err("rocm_xio: failed to query lba size: %s\n",
			fio_rocm_xio_last_error(data->ctx));
		fio_rocm_xio_destroy_ctx(data->ctx);
		data->ctx = NULL;
		return 1;
	}

	for_each_file(td, f, i) {
		if (!f->real_file_size)
			f->real_file_size = td->o.size ? td->o.size : (1ULL << 40);
	}

	return 0;
}

static void fio_rocm_xio_cleanup(struct thread_data *td)
{
	struct rocm_xio_data *data = td->io_ops_data;

	if (!data)
		return;

	fio_rocm_xio_destroy_ctx(data->ctx);
	free(data);
	td->io_ops_data = NULL;
}

static enum fio_q_status fio_rocm_xio_queue(struct thread_data *td,
					    struct io_u *io_u)
{
	struct rocm_xio_data *data = td->io_ops_data;
	int ret;

	switch (io_u->ddir) {
	case DDIR_SYNC:
	case DDIR_DATASYNC:
		io_u->error = 0;
		return FIO_Q_COMPLETED;
	case DDIR_READ:
	case DDIR_WRITE:
		break;
	default:
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "xfer");
		return FIO_Q_COMPLETED;
	}

	if (!data || !data->ctx || !data->lba_size) {
		io_u->error = EIO;
		td_verror(td, io_u->error, "xfer");
		return FIO_Q_COMPLETED;
	}

	if ((io_u->offset % data->lba_size) || (io_u->xfer_buflen % data->lba_size)) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "xfer");
		return FIO_Q_COMPLETED;
	}

	fio_ro_check(td, io_u);

	ret = fio_rocm_xio_submit(data->ctx, io_u->ddir == DDIR_WRITE,
				  io_u->offset, io_u->xfer_buflen);
	if (ret < 0) {
		io_u->error = EIO;
		log_err("rocm_xio: I/O submit failed at off=%llu len=%llu: %s\n",
			(unsigned long long) io_u->offset, io_u->xfer_buflen,
			fio_rocm_xio_last_error(data->ctx));
		td_verror(td, io_u->error, "xfer");
		return FIO_Q_COMPLETED;
	}

	io_u->error = 0;
	return FIO_Q_COMPLETED;
}

static int fio_rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	return 0;
}

static int fio_rocm_xio_invalidate(struct thread_data *td, struct fio_file *f)
{
	return 0;
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name			= "rocm_xio",
	.version		= FIO_IOOPS_VERSION,
	.flags			= FIO_SYNCIO | FIO_DISKLESSIO | FIO_NOEXTEND,
	.setup			= fio_rocm_xio_setup,
	.init			= fio_rocm_xio_init,
	.cleanup		= fio_rocm_xio_cleanup,
	.queue			= fio_rocm_xio_queue,
	.open_file		= fio_rocm_xio_open_file,
	.invalidate		= fio_rocm_xio_invalidate,
	.options		= options,
	.option_struct_size	= sizeof(struct rocm_xio_options),
};

static void fio_init fio_rocm_xio_register(void)
{
	register_ioengine(&ioengine);
}

static void fio_exit fio_rocm_xio_unregister(void)
{
	unregister_ioengine(&ioengine);
}
