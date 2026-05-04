/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../fio.h"
#include "../file.h"
#include "../io_u.h"
#include "../optgroup.h"
#include "rocm_xio_bridge.h"

struct rocm_xio_options {
	void *pad;
	char *controller;
	char *access_pattern;
	uint32_t queue_id;
	uint32_t queue_length;
	uint32_t num_queues;
	uint32_t memory_mode;
	uint32_t nsid;
	uint64_t base_lba;
	uint32_t lbas_per_io;
	uint32_t batch_size;
	uint32_t use_pci_mmio_bridge;
};

static struct fio_option options[] = {
	{
		.name	  = "rocm_xio_controller",
		.lname	  = "ROCm-XIO NVMe controller path",
		.type	  = FIO_OPT_STR_STORE,
		.off1	  = offsetof(struct rocm_xio_options, controller),
		.help	  = "NVMe controller path, such as /dev/nvme0",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_access_pattern",
		.lname	  = "ROCm-XIO access pattern",
		.type	  = FIO_OPT_STR,
		.off1	  = offsetof(struct rocm_xio_options, access_pattern),
		.help	  = "Access pattern for endpoint IO generation",
		.def	  = "random",
		.posval	  = {
			{ .ival = "random", .help = "Random LBA selection" },
			{ .ival = "sequential", .help = "Sequential LBAs" },
		},
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_queue_id",
		.lname	  = "ROCm-XIO queue id",
		.type	  = FIO_OPT_INT,
		.off1	  = offsetof(struct rocm_xio_options, queue_id),
		.help	  = "IO queue id (0 means auto-detect)",
		.def	  = "0",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_queue_length",
		.lname	  = "ROCm-XIO queue depth",
		.type	  = FIO_OPT_INT,
		.off1	  = offsetof(struct rocm_xio_options, queue_length),
		.help	  = "Queue length in entries",
		.def	  = "64",
		.alias	  = "rocm_xio_queue_len",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_num_queues",
		.lname	  = "ROCm-XIO number of queues",
		.type	  = FIO_OPT_INT,
		.off1	  = offsetof(struct rocm_xio_options, num_queues),
		.help	  = "Number of queue pairs to create",
		.def	  = "1",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_memory_mode",
		.lname	  = "ROCm-XIO memory mode bitfield",
		.type	  = FIO_OPT_INT,
		.off1	  = offsetof(struct rocm_xio_options, memory_mode),
		.help	  = "XIO memory mode (0..15)",
		.def	  = "0",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_nsid",
		.lname	  = "ROCm-XIO namespace id",
		.type	  = FIO_OPT_INT,
		.off1	  = offsetof(struct rocm_xio_options, nsid),
		.help	  = "Namespace ID to target",
		.def	  = "1",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_base_lba",
		.lname	  = "ROCm-XIO starting LBA",
		.type	  = FIO_OPT_STR_VAL,
		.off1	  = offsetof(struct rocm_xio_options, base_lba),
		.help	  = "Base LBA for generated accesses",
		.def	  = "0",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_lbas_per_io",
		.lname	  = "ROCm-XIO LBAs per IO",
		.type	  = FIO_OPT_INT,
		.off1	  = offsetof(struct rocm_xio_options, lbas_per_io),
		.help	  = "Number of LBAs in each command",
		.def	  = "1",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_batch_size",
		.lname	  = "ROCm-XIO batch size",
		.type	  = FIO_OPT_INT,
		.off1	  = offsetof(struct rocm_xio_options, batch_size),
		.help	  = "SQE batch size per doorbell (0 means endpoint default)",
		.def	  = "1",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	  = "rocm_xio_pci_mmio_bridge",
		.lname	  = "ROCm-XIO PCI MMIO bridge mode",
		.type	  = FIO_OPT_BOOL,
		.off1	  = offsetof(struct rocm_xio_options, use_pci_mmio_bridge),
		.help	  = "Enable PCI MMIO bridge doorbell route",
		.def	  = "0",
		.category = FIO_OPT_C_ENGINE,
		.group	  = FIO_OPT_G_ROCM_XIO,
	},
	{
		.name = NULL,
	},
};

static int fio_rocm_xio_get_ops(struct thread_data *td, uint32_t *read_io,
				uint32_t *write_io)
{
	uint64_t ios;
	uint32_t r = 0;
	uint32_t w = 0;

	if (!td || !read_io || !write_io)
		return EINVAL;

	if (td_rw(td)) {
		/*
		 * fio uses one io_u per operation; map total_io_u to read+write
		 * counts and split by rwmixread in mixed mode.
		 */
		ios = td->o.number_ios;
		if (!ios)
			ios = 1;
		if (ios > UINT32_MAX)
			ios = UINT32_MAX;

		if (td->o.td_ddir == TD_DDIR_RANDRW || td->o.td_ddir == TD_DDIR_RW) {
			r = (uint32_t)((ios * td->o.rwmix[DDIR_READ]) / 100ULL);
			if (r > ios)
				r = (uint32_t)ios;
			w = (uint32_t)ios - r;
		} else if (td_read(td)) {
			r = (uint32_t)ios;
		} else if (td_write(td)) {
			w = (uint32_t)ios;
		}
	}

	if (!r && !w) {
		if (td_read(td))
			r = 1;
		else if (td_write(td))
			w = 1;
	}

	*read_io = r;
	*write_io = w;
	return 0;
}

static int fio_rocm_xio_prepare_job(struct thread_data *td,
				    struct rocm_xio_options *o,
				    struct rocm_xio_submit_opts *opts)
{
	struct fio_file *f;
	unsigned int i;
	int ret;
	uint32_t read_io = 0;
	uint32_t write_io = 0;
	const char *controller;
	const char *pattern;
	uint32_t queue_id;
	uint32_t queue_len;
	uint32_t num_queues;
	uint32_t memory_mode;
	uint32_t nsid;
	uint64_t base_lba;
	uint32_t lbas_per_io;
	uint32_t batch_size;

	if (!td || !o || !opts)
		return EINVAL;

	memset(opts, 0, sizeof(*opts));

	controller = o->controller;
	if (!controller || !controller[0]) {
		f = NULL;
		for_each_file(td, f, i) {
			if (f && f->file_name && f->file_name[0]) {
				controller = f->file_name;
				break;
			}
		}
	}
	if (!controller || !controller[0]) {
		log_err("rocm-xio: set rocm_xio_controller or filename=/dev/nvmeX\n");
		return EINVAL;
	}

	pattern = o->access_pattern ? o->access_pattern : "random";
	if (strcmp(pattern, "random") && strcmp(pattern, "sequential")) {
		log_err("rocm-xio: rocm_xio_access_pattern must be random|sequential\n");
		return EINVAL;
	}

	queue_id = o->queue_id;
	queue_len = o->queue_length ? o->queue_length : 64;
	num_queues = o->num_queues ? o->num_queues : 1;
	memory_mode = o->memory_mode;
	nsid = o->nsid ? o->nsid : 1;
	base_lba = o->base_lba;
	lbas_per_io = o->lbas_per_io ? o->lbas_per_io : 1;
	batch_size = o->batch_size;

	ret = fio_rocm_xio_get_ops(td, &read_io, &write_io);
	if (ret)
		return ret;

	if (!read_io && !write_io) {
		log_err("rocm-xio: derived zero IO ops, set size/numberio\n");
		return EINVAL;
	}

	opts->controller = controller;
	opts->base_lba = base_lba;
	opts->lbas_per_io = lbas_per_io;
	opts->nsid = nsid;
	opts->queue_id = (uint16_t)queue_id;
	opts->queue_length = (uint16_t)queue_len;
	opts->num_queues = (uint16_t)num_queues;
	opts->memory_mode = memory_mode;
	opts->batch_size = batch_size;
	opts->lfsr_seed = 0;
	opts->use_pci_mmio_bridge = o->use_pci_mmio_bridge;
	opts->verify = 0;
	opts->random_access = !strcmp(pattern, "random");
	opts->do_read = read_io;
	opts->do_write = write_io;

	return 0;
}

static int fio_rocm_xio_init(struct thread_data *td)
{
	(void)td;
	return 0;
}

static int fio_rocm_xio_get_file_size(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_options *o = td->eo;
	uint64_t capacity_lbas = 0;
	uint32_t lba_size = 0;
	uint32_t nsid = 1;
	const char *controller;
	int ret;

	controller = (o && o->controller && o->controller[0]) ?
			o->controller : f->file_name;
	if (!controller || !controller[0])
		return EINVAL;

	if (fio_file_size_known(f))
		return 0;

	if (o && o->nsid)
		nsid = o->nsid;

	ret = rocm_xio_query_lba_size(controller, nsid, &lba_size,
				      NULL, 0);
	if (ret) {
		log_err("rocm-xio: failed to query LBA size on %s (%d)\n",
			controller, ret);
		return ret;
	}

	ret = rocm_xio_query_namespace_capacity(controller, nsid,
						&capacity_lbas, NULL, 0);
	if (ret) {
		log_err("rocm-xio: failed to query namespace capacity on %s (%d)\n",
			controller, ret);
		return ret;
	}

	f->real_file_size = capacity_lbas * lba_size;
	fio_file_set_size_known(f);
	return 0;
}

static enum fio_q_status fio_rocm_xio_queue(struct thread_data *td,
					    struct io_u *io_u)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_submit_opts opts;
	uint32_t verify_pass = 0;
	uint32_t verify_fail = 0;
	uint32_t lba_size = 0;
	const char *controller;
	uint64_t lba_off;
	int ret;

	if (!o) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "rocm-xio options missing");
		return FIO_Q_COMPLETED;
	}

	controller = (o->controller && o->controller[0]) ? o->controller :
		     (io_u->file ? io_u->file->file_name : NULL);
	if (!controller || !controller[0]) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "rocm-xio controller path missing");
		return FIO_Q_COMPLETED;
	}

	switch (io_u->ddir) {
	case DDIR_READ:
	case DDIR_WRITE:
		break;
	case DDIR_SYNC:
	case DDIR_DATASYNC:
	case DDIR_SYNCFS:
		return FIO_Q_COMPLETED;
	default:
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "rocm-xio unsupported ddir");
		return FIO_Q_COMPLETED;
	}

	ret = fio_rocm_xio_prepare_job(td, o, &opts);
	if (ret) {
		io_u->error = ret > 0 ? ret : EINVAL;
		td_verror(td, io_u->error, "rocm-xio setup failed");
		return FIO_Q_COMPLETED;
	}

	opts.controller = controller;
	opts.do_read = (io_u->ddir == DDIR_READ) ? 1 : 0;
	opts.do_write = (io_u->ddir == DDIR_WRITE) ? 1 : 0;
	opts.verify = 0;
	opts.batch_size = o->batch_size ? o->batch_size : 1;

	ret = rocm_xio_query_lba_size(controller, opts.nsid, &lba_size, NULL, 0);
	if (ret || !lba_size) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "rocm-xio failed to query LBA size");
		return FIO_Q_COMPLETED;
	}

	if (io_u->xfer_buflen % lba_size) {
		io_u->error = EINVAL;
		td_verror(td, io_u->error, "rocm-xio xfer size not LBA aligned");
		return FIO_Q_COMPLETED;
	}

	lba_off = io_u->offset / lba_size;
	opts.base_lba += lba_off;
	opts.lbas_per_io = io_u->xfer_buflen / lba_size;
	if (!opts.lbas_per_io)
		opts.lbas_per_io = 1;

	ret = rocm_xio_submit(&opts, &verify_pass, &verify_fail, NULL, 0);
	if (ret) {
		io_u->error = ret > 0 ? ret : EIO;
		td_verror(td, io_u->error, "rocm-xio submit failed");
	}

	return FIO_Q_COMPLETED;
}

static void fio_rocm_xio_cleanup(struct thread_data *td)
{
	(void)td;
}

static int fio_rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	(void) td;
	(void) f;
	return 0;
}

static int fio_rocm_xio_close_file(struct thread_data *td, struct fio_file *f)
{
	(void) td;
	(void) f;
	return 0;
}

static struct ioengine_ops ioengine = {
	.name			= "rocm-xio",
	.version		= FIO_IOOPS_VERSION,
	.init			= fio_rocm_xio_init,
	.queue			= fio_rocm_xio_queue,
	.cleanup		= fio_rocm_xio_cleanup,
	.open_file		= fio_rocm_xio_open_file,
	.close_file		= fio_rocm_xio_close_file,
	.get_file_size		= fio_rocm_xio_get_file_size,
	.flags			= FIO_SYNCIO,
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
