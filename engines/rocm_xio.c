/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../fio.h"
#include "../diskutil.h"
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
	unsigned int inflight;
	unsigned int posted_head;
	unsigned int posted_tail;
	unsigned int lba_size;
	uint64_t capacity_bytes;
	char *controller;
	unsigned int nsid;
};

static const char *fio_rocm_xio_path_basename(const char *path)
{
	const char *base;

	base = strrchr(path, '/');
	return base ? base + 1 : path;
}

static int fio_rocm_xio_namespace_controller(const char *name,
					     char *controller,
					     size_t controller_len,
					     unsigned int *nsid)
{
	const char *p, *n;
	char *end;
	unsigned long id;

	if (strncmp(name, "nvme", 4))
		return -EINVAL;

	p = name + 4;
	if (*p < '0' || *p > '9')
		return -EINVAL;
	while (*p >= '0' && *p <= '9')
		p++;
	if (*p != 'n')
		return -EINVAL;
	n = p;
	p++;
	if (*p < '0' || *p > '9')
		return -EINVAL;

	errno = 0;
	id = strtoul(p, &end, 10);
	if (errno || !id || id > UINT_MAX)
		return -EINVAL;
	p = end;
	while (*p >= '0' && *p <= '9')
		p++;
	if (*p && *p != 'p')
		return -EINVAL;

	if (controller &&
	    snprintf(controller, controller_len, "/dev/%.*s",
		     (int)(n - name), name) >= controller_len)
		return -ENAMETOOLONG;
	if (nsid)
		*nsid = id;

	return 0;
}

static int fio_rocm_xio_find_nvme_controller(const char *sysfs_path,
					     char *controller,
					     size_t controller_len,
					     unsigned int *nsid,
					     unsigned int depth)
{
	char slaves[PATH_MAX], found[PATH_MAX] = { 0 };
	struct dirent *dirent;
	unsigned int found_nr = 0;
	unsigned int found_nsid = 0;
	const char *name;
	DIR *dir;
	int ret;

	if (depth > 8)
		return -ELOOP;

	name = fio_rocm_xio_path_basename(sysfs_path);
	ret = fio_rocm_xio_namespace_controller(name, controller,
						controller_len, nsid);
	if (!ret)
		return 0;

	if (strlen(sysfs_path) > sizeof(slaves) - sizeof("/slaves"))
		return -ENAMETOOLONG;
	snprintf(slaves, sizeof(slaves), "%s/slaves", sysfs_path);
	dir = opendir(slaves);
	if (!dir)
		return -ENODEV;

	ret = 0;
	while ((dirent = readdir(dir)) != NULL) {
		char slave_path[PATH_MAX], candidate[PATH_MAX];
		unsigned int candidate_nsid = 0;
		size_t slaves_len, name_len;

		if (!strcmp(dirent->d_name, ".") ||
		    !strcmp(dirent->d_name, ".."))
			continue;

		slaves_len = strlen(slaves);
		name_len = strlen(dirent->d_name);
		if (slaves_len + 1 + name_len >= sizeof(slave_path)) {
			ret = -ENAMETOOLONG;
			break;
		}
		memcpy(slave_path, slaves, slaves_len);
		slave_path[slaves_len] = '/';
		memcpy(slave_path + slaves_len + 1, dirent->d_name,
		       name_len + 1);

		ret = fio_rocm_xio_find_nvme_controller(slave_path, candidate,
							sizeof(candidate),
							&candidate_nsid,
							depth + 1);
		if (ret)
			continue;
		if (found_nr &&
		    (strcmp(found, candidate) || found_nsid != candidate_nsid)) {
			ret = -EINVAL;
			break;
		}
		snprintf(found, sizeof(found), "%s", candidate);
		found_nsid = candidate_nsid;
		found_nr++;
	}
	closedir(dir);

	if (ret && ret != -ENODEV)
		return ret;
	if (!found_nr)
		return -ENODEV;
	if (snprintf(controller, controller_len, "%s", found) >= controller_len)
		return -ENAMETOOLONG;
	if (nsid)
		*nsid = found_nsid;

	return 0;
}

static int fio_rocm_xio_resolve_controller(struct thread_data *td,
					   char **controller,
					   unsigned int *nsid)
{
	struct fio_file *f;
	char sysfs_path[PATH_MAX], resolved[PATH_MAX];
	unsigned int resolved_nsid = 0;
	int ret;

	if (!td->files_index || !td->files[0] || !td->files[0]->file_name) {
		log_err("rocm_xio: filename is required when rocm_xio_controller or rocm_xio_nsid is auto-detected\n");
		return 1;
	}

	f = td->files[0];
	ret = fio_lookup_block_device(f->file_name, sysfs_path,
				      sizeof(sysfs_path));
	if (ret) {
		log_err("rocm_xio: failed to resolve backing block device for %s\n",
			f->file_name);
		return 1;
	}

	ret = fio_rocm_xio_find_nvme_controller(sysfs_path, resolved,
						sizeof(resolved),
						&resolved_nsid, 0);
	if (ret) {
		log_err("rocm_xio: %s is backed by %s, not an NVMe namespace\n",
			f->file_name, fio_rocm_xio_path_basename(sysfs_path));
		return 1;
	}

	if (controller) {
		*controller = strdup(resolved);
		if (!*controller)
			return 1;
	}
	if (nsid)
		*nsid = resolved_nsid;

	return 0;
}

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

	if (td->o.numjobs != 1) {
		log_err("rocm_xio: numjobs=%u is not supported because cloned jobs share one rocm_xio_queue_id; use one job section with numjobs=1 for each NVMe queue\n",
			td->o.numjobs);
		return 1;
	}

	if (!td->o.use_thread) {
		log_err("rocm_xio: thread=1 is required; process-based jobs cannot safely share ROCm/HIP state\n");
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
		.help	= "NVMe controller path, e.g. /dev/nvme0; auto-detected from filename if omitted",
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
		.def	= "0",
		.help	= "NVMe namespace id, 0 auto-detects from filename",
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

	if ((!((struct rocm_xio_options *)td->eo)->controller ||
	     !((struct rocm_xio_options *)td->eo)->nsid) &&
	    fio_rocm_xio_resolve_controller(td,
			((struct rocm_xio_options *)td->eo)->controller ?
			NULL : &data->controller,
			((struct rocm_xio_options *)td->eo)->nsid ?
			NULL : &data->nsid))
		goto err;

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
		goto err;

	for_each_file(td, f, i) {
		if (generic_get_file_size(td, f))
			goto err;
	}

	return 0;

err:
	free(data->queued);
	free(data->queued_bytes);
	free(data->events);
	free(data->event_gpu_ns);
	free(data->event_bytes);
	free(data->posted_bytes);
	free(data->event_nvme_status);
	free(data->controller);
	free(data);
	td->io_ops_data = NULL;
	return 1;
}

static int fio_rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_data *data = td->io_ops_data;
	struct fio_rocm_xio_session_opts opts = { 0 };
	struct fio_rocm_xio_namespace_info ns = { 0 };
	struct fio_file *f;
	const char *controller = o->controller;
	unsigned int nsid = o->nsid;
	int i;

	if (!controller) {
		if (!data->controller &&
		    fio_rocm_xio_resolve_controller(td, &data->controller,
						    nsid ? NULL : &data->nsid))
			return 1;
		controller = data->controller;
	}
	if (!nsid) {
		if (!data->nsid &&
		    fio_rocm_xio_resolve_controller(td,
						    controller ? NULL : &data->controller,
						    &data->nsid))
			return 1;
		nsid = data->nsid;
	}

	opts.controller = controller;
	if (td->files_index)
		opts.filename = td->files[0]->file_name;
	opts.queue_id = o->queue_id;
	opts.queue_length = o->queue_length;
	opts.nsid = nsid;
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
	free(data->controller);
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
		if (ret == -EAGAIN)
			break;
		if (ret < 0) {
			io_u->error = -ret;
			data->event_gpu_ns[data->events_nr] = 0;
			data->event_bytes[data->events_nr] = data->queued_bytes[i];
			data->event_nvme_status[data->events_nr] = 0;
			data->events[data->events_nr++] = io_u;
		} else {
			data->posted_bytes[data->posted_tail++ % td->o.iodepth] =
				data->queued_bytes[i];
			data->inflight++;
			io_u_queued(td, io_u);
			posted++;
		}
	}

	if (posted)
		io_u_mark_submit(td, posted);
	if (i < data->queued_nr) {
		unsigned int left = data->queued_nr - i;

		memmove(data->queued, data->queued + i,
			left * sizeof(*data->queued));
		memmove(data->queued_bytes, data->queued_bytes + i,
			left * sizeof(*data->queued_bytes));
		data->queued_nr = left;
	} else {
		data->queued_nr = 0;
	}
	return 0;
}

static int fio_rocm_xio_getevents(struct thread_data *td, unsigned int min,
				  unsigned int max,
				  const struct timespec fio_unused *t)
{
	struct rocm_xio_data *data = td->io_ops_data;
	struct timespec now;
	enum { rocm_xio_reap_batch = 32 };

	if (!data)
		return 0;

	if (!data->events_nr && data->queued_nr)
		fio_rocm_xio_commit(td);

	while (data->events_nr < max) {
		struct fio_rocm_xio_completion comps[rocm_xio_reap_batch] = { 0 };
		unsigned int want = max - data->events_nr;
		int ret;

		if (want > rocm_xio_reap_batch)
			want = rocm_xio_reap_batch;

		ret = fio_rocm_xio_reap(data->ctx, 0, want, comps, NULL);
		if (ret <= 0) {
			if (data->events_nr < min)
				sched_yield();
			else
				break;
			continue;
		}

		for (int i = 0; i < ret && data->events_nr < max; i++) {
			struct fio_rocm_xio_completion *comp = &comps[i];
			struct io_u *io_u =
				(struct io_u *)(uintptr_t) comp->user_data;

			if (io_u) {
				io_u->error = comp->error;
				io_u->resid = comp->error ? io_u->xfer_buflen : 0;
			}
			data->event_gpu_ns[data->events_nr] = comp->gpu_elapsed_ns;
			data->event_bytes[data->events_nr] =
				data->posted_bytes[data->posted_head++ % td->o.iodepth];
			if (data->inflight)
				data->inflight--;
			data->event_nvme_status[data->events_nr] = comp->nvme_status;
			if (io_u && comp->gpu_elapsed_ns) {
				uint64_t gpu_ns = comp->gpu_elapsed_ns;

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
	if (io_u && io_u->error)
		log_err("rocm_xio: completion error=%d, NVMe status=0x%04x, offset=%llu, len=%llu\n",
			io_u->error, data->event_nvme_status[event],
			(unsigned long long) io_u->offset,
			(unsigned long long) io_u->xfer_buflen);

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

	if (data->events_nr || data->queued_nr + data->inflight >= td->o.iodepth)
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
