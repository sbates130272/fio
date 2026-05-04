/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

/*
 * rocm-xio engine
 *
 * Synchronous fio engine that delegates NVMe I/O to the ROCm XIO nvme-ep
 * endpoint through xio-tester. Each fio I/O becomes one GPU-initiated NVMe
 * operation with the fio offset translated to a base LBA.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../fio.h"
#include "../optgroup.h"

#define ROCM_XIO_DEFAULT_TESTER "xio-tester"
#define ROCM_XIO_DEFAULT_LBA_SIZE 512U
#define ROCM_XIO_DEFAULT_QUEUE_LENGTH 64U
#define ROCM_XIO_DEFAULT_NUM_QUEUES 1U
#define ROCM_XIO_DEFAULT_BATCH_SIZE 1U
#define ROCM_XIO_DEFAULT_BUFFER_SIZE (1024ULL * 1024ULL)
#define ROCM_XIO_MAX_ARGS 64
#define ROCM_XIO_ARG_BUFLEN 48

struct rocm_xio_options {
	void *pad;
	char *tester;
	char *controller;
	unsigned int nsid;
	unsigned int lba_size;
	unsigned int queue_id;
	unsigned int queue_length;
	unsigned int num_queues;
	unsigned int batch_size;
	unsigned int memory_mode;
	unsigned int pci_mmio_bridge;
	unsigned int verbose;
	unsigned int lfsr_seed;
	unsigned long long data_buffer_size;
};

struct rocm_xio_data {
	const char *controller;
};

struct rocm_xio_argv {
	char *argv[ROCM_XIO_MAX_ARGS];
	char storage[ROCM_XIO_MAX_ARGS][ROCM_XIO_ARG_BUFLEN];
	unsigned int argc;
};

static struct fio_option options[] = {
	{
		.name	= "rocm_xio_tester",
		.lname	= "ROCm XIO tester path",
		.type	= FIO_OPT_STR_STORE,
		.off1	= offsetof(struct rocm_xio_options, tester),
		.help	= "Path to xio-tester from the rocm-xio project",
		.def	= ROCM_XIO_DEFAULT_TESTER,
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_controller",
		.lname	= "ROCm XIO NVMe controller",
		.type	= FIO_OPT_STR_STORE,
		.off1	= offsetof(struct rocm_xio_options, controller),
		.help	= "NVMe controller path passed to xio-tester",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_nsid",
		.lname	= "ROCm XIO namespace ID",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, nsid),
		.help	= "NVMe namespace ID",
		.def	= "1",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_lba_size",
		.lname	= "ROCm XIO LBA size",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, lba_size),
		.help	= "LBA size used to translate fio offsets and lengths",
		.def	= "512",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_queue_id",
		.lname	= "ROCm XIO queue ID",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, queue_id),
		.help	= "NVMe queue ID, or 0 for rocm-xio auto-detection",
		.def	= "0",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_queue_length",
		.lname	= "ROCm XIO queue length",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, queue_length),
		.help	= "NVMe queue length in entries",
		.def	= "64",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_num_queues",
		.lname	= "ROCm XIO queue count",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, num_queues),
		.help	= "Number of independent NVMe queues",
		.def	= "1",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_batch_size",
		.lname	= "ROCm XIO batch size",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, batch_size),
		.help	= "Number of SQEs submitted per doorbell in xio-tester",
		.def	= "1",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_memory_mode",
		.lname	= "ROCm XIO memory mode",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, memory_mode),
		.help	= "rocm-xio memory mode bitmask",
		.def	= "0",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_pci_mmio_bridge",
		.lname	= "ROCm XIO PCI MMIO bridge",
		.type	= FIO_OPT_STR_SET,
		.off1	= offsetof(struct rocm_xio_options, pci_mmio_bridge),
		.help	= "Use the rocm-xio PCI MMIO bridge doorbell path",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_verbose",
		.lname	= "ROCm XIO verbose output",
		.type	= FIO_OPT_STR_SET,
		.off1	= offsetof(struct rocm_xio_options, verbose),
		.help	= "Preserve xio-tester stdout and pass -v",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_lfsr_seed",
		.lname	= "ROCm XIO LFSR seed",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, lfsr_seed),
		.help	= "LFSR seed for rocm-xio write pattern generation",
		.def	= "0",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_data_buffer_size",
		.lname	= "ROCm XIO data buffer size",
		.type	= FIO_OPT_ULL,
		.off1	= offsetof(struct rocm_xio_options, data_buffer_size),
		.help	= "Data buffer size passed to xio-tester",
		.def	= "1048576",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= NULL,
	},
};

static int rocm_xio_push(struct rocm_xio_argv *cmd, const char *arg)
{
	if (cmd->argc + 1 >= ROCM_XIO_MAX_ARGS)
		return -E2BIG;

	cmd->argv[cmd->argc++] = (char *) arg;
	cmd->argv[cmd->argc] = NULL;
	return 0;
}

static int rocm_xio_pushf(struct rocm_xio_argv *cmd, const char *fmt, ...)
{
	va_list ap;
	int ret;

	if (cmd->argc + 1 >= ROCM_XIO_MAX_ARGS)
		return -E2BIG;

	va_start(ap, fmt);
	ret = vsnprintf(cmd->storage[cmd->argc],
			sizeof(cmd->storage[cmd->argc]), fmt, ap);
	va_end(ap);

	if (ret < 0 || (size_t) ret >= sizeof(cmd->storage[cmd->argc]))
		return -ENAMETOOLONG;

	return rocm_xio_push(cmd, cmd->storage[cmd->argc]);
}

static int rocm_xio_wait_child(pid_t pid)
{
	int status;

	while (waitpid(pid, &status, 0) < 0) {
		if (errno != EINTR)
			return -errno;
	}

	if (WIFEXITED(status))
		return WEXITSTATUS(status) ? -EIO : 0;
	if (WIFSIGNALED(status))
		return -EINTR;

	return -EIO;
}

static int rocm_xio_exec(struct thread_data *td, struct rocm_xio_argv *cmd,
			 bool verbose)
{
	pid_t pid;

	pid = fork();
	if (pid < 0)
		return -errno;

	if (!pid) {
		if (!verbose) {
			int nullfd = open("/dev/null", O_WRONLY);

			if (nullfd >= 0) {
				dup2(nullfd, STDOUT_FILENO);
				close(nullfd);
			}
		}

		execvp(cmd->argv[0], cmd->argv);
		log_err("rocm-xio: execvp(%s) failed: %s\n", cmd->argv[0],
			strerror(errno));
		_exit(127);
	}

	return rocm_xio_wait_child(pid);
}

static int rocm_xio_build_cmd(struct thread_data *td, struct io_u *io_u,
			      struct rocm_xio_argv *cmd)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_data *xd = td->io_ops_data;
	uint64_t base_lba;
	uint64_t lbas_per_io;
	int ret;

	if (io_u->ddir != DDIR_READ && io_u->ddir != DDIR_WRITE) {
		io_u->error = EINVAL;
		return -EINVAL;
	}

	if (!o->lba_size)
		o->lba_size = ROCM_XIO_DEFAULT_LBA_SIZE;
	if (!o->queue_length)
		o->queue_length = ROCM_XIO_DEFAULT_QUEUE_LENGTH;
	if (!o->num_queues)
		o->num_queues = ROCM_XIO_DEFAULT_NUM_QUEUES;
	if (!o->batch_size)
		o->batch_size = ROCM_XIO_DEFAULT_BATCH_SIZE;
	if (!o->data_buffer_size)
		o->data_buffer_size = ROCM_XIO_DEFAULT_BUFFER_SIZE;

	if (io_u->offset % o->lba_size || io_u->xfer_buflen % o->lba_size) {
		log_err("rocm-xio: offset and length must be aligned to "
			"rocm_xio_lba_size=%u\n", o->lba_size);
		io_u->error = EINVAL;
		return -EINVAL;
	}

	base_lba = io_u->offset / o->lba_size;
	lbas_per_io = io_u->xfer_buflen / o->lba_size;
	if (!lbas_per_io) {
		io_u->error = EINVAL;
		return -EINVAL;
	}

	memset(cmd, 0, sizeof(*cmd));

	ret = rocm_xio_push(cmd, o->tester ? o->tester : ROCM_XIO_DEFAULT_TESTER);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--memory-mode");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", o->memory_mode);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--less-timing");
	if (ret)
		return ret;
	if (o->pci_mmio_bridge) {
		ret = rocm_xio_push(cmd, "--pci-mmio-bridge");
		if (ret)
			return ret;
	}
	if (o->verbose) {
		ret = rocm_xio_push(cmd, "-v");
		if (ret)
			return ret;
	}
	ret = rocm_xio_push(cmd, "nvme-ep");
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--controller");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%s", xd->controller);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--access-pattern");
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "sequential");
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--namespace");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", o->nsid);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--base-lba");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%llu", (unsigned long long) base_lba);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--lbas-per-io");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%llu", (unsigned long long) lbas_per_io);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--read-io");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", io_u->ddir == DDIR_READ ? 1 : 0);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--write-io");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", io_u->ddir == DDIR_WRITE ? 1 : 0);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--queue-id");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", o->queue_id);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--queue-length");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", o->queue_length);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--num-queues");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", o->num_queues);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--batch-size");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", o->batch_size);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--lfsr-seed");
	if (ret)
		return ret;
	ret = rocm_xio_pushf(cmd, "%u", o->lfsr_seed);
	if (ret)
		return ret;
	ret = rocm_xio_push(cmd, "--data-buffer-size");
	if (ret)
		return ret;
	return rocm_xio_pushf(cmd, "%llu", o->data_buffer_size);
}

static enum fio_q_status rocm_xio_queue(struct thread_data *td,
					struct io_u *io_u)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_argv cmd;
	int ret;

	fio_ro_check(td, io_u);

	ret = rocm_xio_build_cmd(td, io_u, &cmd);
	if (!ret)
		ret = rocm_xio_exec(td, &cmd, o->verbose);

	if (ret) {
		io_u->error = -ret;
		td_verror(td, io_u->error, "rocm-xio I/O");
	}

	return FIO_Q_COMPLETED;
}

static int rocm_xio_open(struct thread_data *td, struct fio_file *f)
{
	struct rocm_xio_options *o = td->eo;
	struct rocm_xio_data *xd = td->io_ops_data;

	if (!xd->controller) {
		if (o->controller)
			xd->controller = o->controller;
		else if (f->file_name)
			xd->controller = f->file_name;
	}

	if (!xd->controller) {
		log_err("rocm-xio: set filename or rocm_xio_controller\n");
		return 1;
	}

	return 0;
}

static int rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_data *xd;

	if (td->o.verify) {
		log_err("rocm-xio: fio verify is not supported because "
			"rocm-xio owns the GPU data buffers\n");
		return 1;
	}

	xd = calloc(1, sizeof(*xd));
	if (!xd)
		return 1;

	td->io_ops_data = xd;
	return 0;
}

static void rocm_xio_cleanup(struct thread_data *td)
{
	free(td->io_ops_data);
	td->io_ops_data = NULL;
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name			= "rocm-xio",
	.version		= FIO_IOOPS_VERSION,
	.init			= rocm_xio_init,
	.queue			= rocm_xio_queue,
	.cleanup		= rocm_xio_cleanup,
	.open_file		= rocm_xio_open,
	.flags			= FIO_SYNCIO | FIO_RAWIO | FIO_NOEXTEND |
				  FIO_NODISKUTIL,
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
