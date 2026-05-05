/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include <errno.h>
#include <stdint.h>
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
	int verify_lfsr;
	int verbose;
};

struct rocm_xio_data {
	struct fio_rocm_xio_session *ctx;
	struct io_u **queued;
	uint64_t *queued_bytes;
	struct io_u **events;
	uint64_t *event_gpu_ns;
	uint64_t *event_bytes;
	uint64_t *posted_bytes;
	uint16_t *event_nvme_status;
	unsigned int queued_nr;
	unsigned int events_nr;
	unsigned int last_events;
	unsigned int posted_head;
	unsigned int posted_tail;
	unsigned int lba_size;
	uint64_t capacity_bytes;
};

static int fio_rocm_xio_validate_options(struct thread_data *td)
{
	struct rocm_xio_options *o = td->eo;
	unsigned int rocm_xio_jobs = 0;

	if (!td->o.iodepth) {
		log_err("rocm_xio: iodepth must be greater than zero\n");
		return 1;
	}

	if (!o->queue_length) {
		log_err("rocm_xio: rocm_xio_queue_length must be greater than zero\n");
		return 1;
	}

	if (!o->batch_size) {
		log_err("rocm_xio: rocm_xio_batch_size must be greater than zero\n");
		return 1;
	}

	if (o->batch_size > td->o.iodepth) {
		log_err("rocm_xio: rocm_xio_batch_size=%u exceeds iodepth=%u\n",
			o->batch_size, td->o.iodepth);
		return 1;
	}

	if (td->o.iodepth > o->queue_length) {
		log_err("rocm_xio: iodepth=%u exceeds rocm_xio_queue_length=%u\n",
			td->o.iodepth, o->queue_length);
		return 1;
	}

	for_each_td(td2) {
		if (td2->io_ops != td->io_ops)
			continue;
		rocm_xio_jobs++;
		if (td2 == td)
			continue;
		if (((struct rocm_xio_options *)td2->eo)->queue_id == o->queue_id &&
		    o->queue_id) {
			log_err("rocm_xio: duplicate rocm_xio_queue_id=%u across jobs\n",
				o->queue_id);
			return 1;
		}
	} end_for_each();

	if (rocm_xio_jobs > 1 && !o->queue_id) {
		log_err("rocm_xio: rocm_xio_queue_id=0 auto-detect is only valid for single-job runs; set a distinct queue ID per job\n");
		return 1;
	}

	return 0;
}

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
		.name	= "rocm_xio_verify_lfsr",
		.lname	= "ROCm XIO LFSR verify",
		.type	= FIO_OPT_BOOL,
		.off1	= offsetof(struct rocm_xio_options, verify_lfsr),
		.def	= "0",
		.help	= "Enable ROCm XIO LFSR pattern verification",
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
	unsigned int i;

	data = calloc(1, sizeof(*data));
	if (!data)
		return 1;

	if (fio_rocm_xio_validate_options(td)) {
		free(data);
		return 1;
	}

	td->io_ops_data = data;

	data->queued = calloc(td->o.iodepth, sizeof(*data->queued));
	data->queued_bytes = calloc(td->o.iodepth, sizeof(*data->queued_bytes));
	data->events = calloc(td->o.iodepth, sizeof(*data->events));
	data->event_gpu_ns = calloc(td->o.iodepth, sizeof(*data->event_gpu_ns));
	data->event_bytes = calloc(td->o.iodepth, sizeof(*data->event_bytes));
	data->posted_bytes = calloc(td->o.iodepth, sizeof(*data->posted_bytes));
	data->event_nvme_status = calloc(td->o.iodepth,
					 sizeof(*data->event_nvme_status));
	if (!data->queued || !data->queued_bytes || !data->events ||
	    !data->event_gpu_ns ||
	    !data->event_bytes || !data->posted_bytes || !data->event_nvme_status)
		return 1;

	for_each_file(td, f, i) {
		if (generic_get_file_size(td, f))
			return 1;
	}

	return 0;
}

static int fio_rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_data *data = td->io_ops_data;
	struct fio_rocm_xio_session_opts opts = { 0 };
	struct fio_rocm_xio_namespace_info ns = { 0 };
	struct fio_file *f;
	int i;

	if (!o->controller) {
		log_err("rocm_xio: rocm_xio_controller must be set\n");
		return 1;
	}

	opts.controller = o->controller;
	if (td->files_index)
		opts.filename = td->files[0]->file_name;
	opts.queue_id = o->queue_id;
	opts.queue_length = o->queue_length;
	opts.nsid = o->nsid;
	opts.lfsr_seed = o->lfsr_seed;
	opts.batch_size = o->batch_size;
	opts.memory_mode = o->memory_mode;
	opts.verify_mode = o->verify_lfsr ? FIO_ROCM_XIO_VERIFY_LFSR :
					   FIO_ROCM_XIO_VERIFY_NONE;
	opts.ring_depth = td->o.iodepth;
	opts.gpu_id = o->gpu_id;
	opts.use_pci_mmio_bridge = o->pci_mmio_bridge;
	opts.verbose = o->verbose;

	if (fio_rocm_xio_open_session(&opts, &data->ctx) < 0 || !data->ctx) {
		log_err("rocm_xio: failed to initialize rocm-xio context\n");
		return 1;
	}

	if (fio_rocm_xio_get_namespace_info(data->ctx, &ns) < 0) {
		log_err("rocm_xio: failed to query lba size: %s\n",
			fio_rocm_xio_last_error(data->ctx));
		fio_rocm_xio_close_session(data->ctx);
		data->ctx = NULL;
		return 1;
	}
	data->lba_size = ns.lba_size;
	data->capacity_bytes = ns.capacity_bytes;

	for_each_file(td, f, i) {
		if (data->capacity_bytes)
			f->real_file_size = data->capacity_bytes;
	}

	return 0;
}

static void fio_rocm_xio_cleanup(struct thread_data *td)
{
	struct rocm_xio_data *data = td->io_ops_data;

	if (!data)
		return;

	fio_rocm_xio_close_session(data->ctx);
	free(data->queued);
	free(data->queued_bytes);
	free(data->events);
	free(data->event_gpu_ns);
	free(data->event_bytes);
	free(data->posted_bytes);
	free(data->event_nvme_status);
	free(data);
	td->io_ops_data = NULL;
}

static int fio_rocm_xio_commit(struct thread_data *td)
{
	struct rocm_xio_data *data = td->io_ops_data;
	unsigned int i, posted = 0;

	if (!data)
		return 1;

	if (data->events_nr)
		return 0;

	for (i = 0; i < data->queued_nr; i++) {
		struct io_u *io_u = data->queued[i];
		struct fio_rocm_xio_io_desc desc = { 0 };
		int ret;

		desc.user_data = (uintptr_t) io_u;
		desc.offset = io_u->offset;
		desc.len = data->queued_bytes[i];
		desc.is_write = io_u->ddir == DDIR_WRITE;
		desc.do_verify = ((struct rocm_xio_options *) td->eo)->verify_lfsr;

		ret = fio_rocm_xio_post_desc(data->ctx, &desc);
		if (ret < 0) {
			io_u->error = EIO;
			data->event_gpu_ns[data->events_nr] = 0;
			data->event_bytes[data->events_nr] = data->queued_bytes[i];
			data->event_nvme_status[data->events_nr] = 0;
			data->events[data->events_nr++] = io_u;
		} else {
			data->posted_bytes[data->posted_tail++ % td->o.iodepth] =
				data->queued_bytes[i];
			io_u_queued(td, io_u);
			posted++;
		}
	}

	if (posted)
		io_u_mark_submit(td, posted);
	data->queued_nr = 0;

	while (posted && data->events_nr < posted) {
		struct fio_rocm_xio_completion comp = { 0 };
		struct io_u *io_u;
		int ret;

		ret = fio_rocm_xio_reap(data->ctx, 1, 1, &comp, NULL);
		if (ret <= 0)
			continue;

		io_u = (struct io_u *)(uintptr_t) comp.user_data;
		if (io_u)
			io_u->error = comp.error;
		data->event_gpu_ns[data->events_nr] = comp.gpu_elapsed_ns;
		data->event_bytes[data->events_nr] =
			data->posted_bytes[data->posted_head++ % td->o.iodepth];
		data->event_nvme_status[data->events_nr] = comp.nvme_status;
		if (io_u && comp.gpu_elapsed_ns) {
			struct timespec now;
			uint64_t gpu_ns = comp.gpu_elapsed_ns;

			fio_gettime(&now, NULL);
			io_u->issue_time = now;
			while (gpu_ns >= 1000000000ULL) {
				io_u->issue_time.tv_sec--;
				gpu_ns -= 1000000000ULL;
			}
			if ((uint64_t) io_u->issue_time.tv_nsec >= gpu_ns) {
				io_u->issue_time.tv_nsec -= gpu_ns;
			} else {
				io_u->issue_time.tv_sec--;
				io_u->issue_time.tv_nsec += 1000000000ULL - gpu_ns;
			}
		}
		data->events[data->events_nr++] = io_u;
	}
	return 0;
}

static int fio_rocm_xio_getevents(struct thread_data *td, unsigned int min,
				  unsigned int max,
				  const struct timespec fio_unused *t)
{
	struct rocm_xio_data *data = td->io_ops_data;
	struct timespec now;

	if (!data)
		return 0;

	if (!data->events_nr && data->queued_nr)
		fio_rocm_xio_commit(td);

	while (data->events_nr < max) {
		struct fio_rocm_xio_completion comp = { 0 };
		struct io_u *io_u;
		int ret;

		ret = fio_rocm_xio_reap(data->ctx, 0, 1, &comp, NULL);
		if (ret <= 0)
			break;
		io_u = (struct io_u *)(uintptr_t) comp.user_data;
		if (io_u) {
			io_u->error = comp.error;
			io_u->resid = comp.error ? io_u->xfer_buflen : 0;
		}
		data->event_gpu_ns[data->events_nr] = comp.gpu_elapsed_ns;
		data->event_bytes[data->events_nr] =
			data->posted_bytes[data->posted_head++ % td->o.iodepth];
		data->event_nvme_status[data->events_nr] = comp.nvme_status;
		if (io_u && comp.gpu_elapsed_ns) {
			uint64_t gpu_ns = comp.gpu_elapsed_ns;

			fio_gettime(&now, NULL);
			io_u->issue_time = now;
			while (gpu_ns >= 1000000000ULL) {
				io_u->issue_time.tv_sec--;
				gpu_ns -= 1000000000ULL;
			}
			if ((uint64_t) io_u->issue_time.tv_nsec >= gpu_ns) {
				io_u->issue_time.tv_nsec -= gpu_ns;
			} else {
				io_u->issue_time.tv_sec--;
				io_u->issue_time.tv_nsec += 1000000000ULL - gpu_ns;
			}
		}
		data->events[data->events_nr++] = io_u;
	}

	if (data->events_nr > max)
		data->last_events = max;
	else
		data->last_events = data->events_nr;
	data->events_nr = 0;
	return data->last_events;
}

static struct io_u *fio_rocm_xio_event(struct thread_data *td, int event)
{
	struct rocm_xio_data *data = td->io_ops_data;
	struct io_u *io_u;

	if (!data || event < 0 || (unsigned int) event >= data->last_events)
		return NULL;

	io_u = data->events[event];
	if (io_u) {
		if (data->event_bytes[event]) {
			io_u->buflen = data->event_bytes[event];
			io_u->xfer_buflen = data->event_bytes[event];
		}
		io_u->resid = io_u->error ? io_u->xfer_buflen : 0;
	}
	if (io_u && io_u->error && data->event_nvme_status[event])
		log_err("rocm_xio: NVMe status=0x%04x\n",
			data->event_nvme_status[event]);

	return io_u;
}

static enum fio_q_status fio_rocm_xio_queue(struct thread_data *td,
					    struct io_u *io_u)
{
	struct rocm_xio_data *data = td->io_ops_data;

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

	if (data->events_nr || data->queued_nr == td->o.iodepth)
		return FIO_Q_BUSY;

	io_u->error = 0;
	data->queued[data->queued_nr] = io_u;
	data->queued_bytes[data->queued_nr] = io_u->xfer_buflen;
	data->queued_nr++;
	return FIO_Q_QUEUED;
}

static int fio_rocm_xio_open_file(struct thread_data *td, struct fio_file *f)
{
	return generic_open_file(td, f);
}

static int fio_rocm_xio_close_file(struct thread_data *td, struct fio_file *f)
{
	return generic_close_file(td, f);
}

static int fio_rocm_xio_get_file_size(struct thread_data *td, struct fio_file *f)
{
	return generic_get_file_size(td, f);
}

static int fio_rocm_xio_invalidate(struct thread_data *td, struct fio_file *f)
{
	return 0;
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name			= "rocm_xio",
	.version		= FIO_IOOPS_VERSION,
	.flags			= FIO_NOEXTEND | FIO_ASYNCIO_SETS_ISSUE_TIME,
	.setup			= fio_rocm_xio_setup,
	.init			= fio_rocm_xio_init,
	.cleanup		= fio_rocm_xio_cleanup,
	.commit			= fio_rocm_xio_commit,
	.getevents		= fio_rocm_xio_getevents,
	.event			= fio_rocm_xio_event,
	.queue			= fio_rocm_xio_queue,
	.open_file		= fio_rocm_xio_open_file,
	.close_file		= fio_rocm_xio_close_file,
	.get_file_size		= fio_rocm_xio_get_file_size,
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
