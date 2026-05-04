/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <algorithm>
#include <memory>
#include <string>

#include <xio.h>
#include <endpoints/nvme-ep/nvme-ep.h>

extern "C" {
#include "../fio.h"
#include "../optgroup.h"
}

struct rocm_xio_options {
	void *padding;
	char *controller;
	unsigned int queue_id;
	unsigned int queue_length;
	unsigned int namespace_id;
	unsigned int memory_mode;
	unsigned int pci_mmio_bridge;
	unsigned int batch_size;
	unsigned int num_queues;
};

struct rocm_xio_data {
	std::unique_ptr<xio::XioEndpoint> endpoint;
	xio::nvme_ep::nvmeEpConfig *endpoint_cfg;
	xio::XioEndpointConfig run_cfg;
	std::string controller_path;
	unsigned int lba_size;
};

static struct fio_option options[] = {
	{
		.name	= "rocm_xio_controller",
		.lname	= "ROCm XIO NVMe controller path",
		.type	= FIO_OPT_STR_STORE,
		.off1	= offsetof(struct rocm_xio_options, controller),
		.help	= "NVMe controller path, e.g. /dev/nvme0",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_queue_id",
		.lname	= "ROCm XIO queue id",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, queue_id),
		.def	= "0",
		.help	= "NVMe queue ID (0 means auto-detect last I/O queue)",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_queue_length",
		.lname	= "ROCm XIO queue length",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, queue_length),
		.def	= "1024",
		.help	= "NVMe queue length (power of two)",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_namespace",
		.lname	= "ROCm XIO namespace id",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, namespace_id),
		.def	= "1",
		.help	= "NVMe namespace ID",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_memory_mode",
		.lname	= "ROCm XIO memory mode",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, memory_mode),
		.def	= "0",
		.help	= "XIO memoryMode bitmask for queues and buffers",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_pci_mmio_bridge",
		.lname	= "ROCm XIO PCI MMIO bridge",
		.type	= FIO_OPT_BOOL,
		.off1	= offsetof(struct rocm_xio_options, pci_mmio_bridge),
		.def	= "0",
		.help	= "Use PCI MMIO bridge doorbells instead of direct BAR0",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_batch_size",
		.lname	= "ROCm XIO batch size",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, batch_size),
		.def	= "1",
		.help	= "SQEs per doorbell ring for nvme-ep",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= "rocm_xio_num_queues",
		.lname	= "ROCm XIO number of queues",
		.type	= FIO_OPT_INT,
		.off1	= offsetof(struct rocm_xio_options, num_queues),
		.def	= "1",
		.help	= "Number of NVMe queues to use",
		.category = FIO_OPT_C_ENGINE,
		.group	= FIO_OPT_G_ROCM_XIO,
	},
	{
		.name	= NULL,
	},
};

static std::string rocm_xio_get_controller(struct thread_data *td,
					   struct rocm_xio_options *o)
{
	if (o->controller && o->controller[0])
		return std::string(o->controller);

	if (td->o.nr_files && td->files && td->files[0] &&
	    td->files[0]->file_name && td->files[0]->file_name[0])
		return std::string(td->files[0]->file_name);

	if (td->o.filename && td->o.filename[0])
		return std::string(td->o.filename);

	return std::string();
}

static int fio_rocm_xio_init(struct thread_data *td)
{
	struct rocm_xio_options *o = static_cast<struct rocm_xio_options *>(td->eo);
	auto *xd = new rocm_xio_data();
	unsigned int max_bs;
	std::string err;

	if (o->namespace_id == 0) {
		log_err("rocm-xio: rocm_xio_namespace must be > 0\n");
		delete xd;
		return 1;
	}
	if (o->queue_id > UINT16_MAX || o->queue_length == 0 ||
	    o->queue_length > UINT16_MAX || o->num_queues == 0 ||
	    o->num_queues > UINT16_MAX) {
		log_err("rocm-xio: queue_id/queue_length/num_queues out of range\n");
		delete xd;
		return 1;
	}
	if (o->queue_length & (o->queue_length - 1)) {
		log_err("rocm-xio: rocm_xio_queue_length must be a power of two\n");
		delete xd;
		return 1;
	}

	xd->endpoint = xio::createEndpoint("nvme-ep");
	if (!xd->endpoint) {
		log_err("rocm-xio: failed creating nvme-ep endpoint\n");
		delete xd;
		return 1;
	}

	xd->endpoint_cfg = static_cast<xio::nvme_ep::nvmeEpConfig *>(
			xd->endpoint->initializeEndpointConfig());
	if (!xd->endpoint_cfg) {
		log_err("rocm-xio: failed initializing endpoint configuration\n");
		delete xd;
		return 1;
	}

	xd->controller_path = rocm_xio_get_controller(td, o);
	if (xd->controller_path.empty()) {
		log_err("rocm-xio: set rocm_xio_controller or filename=/dev/nvmeX\n");
		delete xd;
		return 1;
	}

	xd->endpoint_cfg->controller = xd->controller_path;
	xd->endpoint_cfg->queueId = static_cast<uint16_t>(o->queue_id);
	xd->endpoint_cfg->queueLength = static_cast<uint16_t>(o->queue_length);
	xd->endpoint_cfg->numQueues = static_cast<uint16_t>(o->num_queues);
	xd->endpoint_cfg->ioParams.nsid = o->namespace_id;
	xd->endpoint_cfg->ioParams.baseLba = 0;
	xd->endpoint_cfg->ioParams.readIo = 1;
	xd->endpoint_cfg->ioParams.writeIo = 0;
	xd->endpoint_cfg->ioParams.lbasPerIo = 1;
	xd->endpoint_cfg->ioParams.batchSize = o->batch_size;
	xd->endpoint_cfg->ioParams.infiniteMode = false;
	xd->endpoint_cfg->ioParams.accessPattern = "sequential";
	xd->endpoint_cfg->verify = false;
	xd->endpoint_cfg->doorbellParams.usePciMmioBridge = o->pci_mmio_bridge;

	max_bs = std::max(td->o.max_bs[DDIR_READ], td->o.max_bs[DDIR_WRITE]);
	if (!max_bs)
		max_bs = 4096;
	xd->endpoint_cfg->bufferParams.bufferSize = max_bs;

	memset(&xd->run_cfg, 0, sizeof(xd->run_cfg));
	xd->run_cfg.iterations = 1;
	xd->run_cfg.numThreads = 1;
	xd->run_cfg.memoryMode = o->memory_mode;
	xd->run_cfg.pciMmioBridge = o->pci_mmio_bridge;
	xd->run_cfg.endpointConfig = xd->endpoint_cfg;

	xd->endpoint->applyCommonConfig(xd->endpoint_cfg, &xd->run_cfg);
	err = xd->endpoint->validateConfig(xd->endpoint_cfg);
	if (!err.empty()) {
		log_err("rocm-xio: invalid configuration: %s\n", err.c_str());
		delete xd;
		return 1;
	}

	xd->lba_size = xd->endpoint_cfg->ioParams.lbaSize;
	if (!xd->lba_size) {
		log_err("rocm-xio: detected invalid LBA size 0\n");
		delete xd;
		return 1;
	}

	td->io_ops_data = xd;
	return 0;
}

static enum fio_q_status fio_rocm_xio_queue(struct thread_data *td,
					    struct io_u *io_u)
{
	auto *xd = static_cast<rocm_xio_data *>(td->io_ops_data);
	hipError_t run_err;
	uint64_t lbas;

	if (!xd || !xd->endpoint_cfg) {
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "rocm-xio queue");
		return FIO_Q_COMPLETED;
	}

	if (io_u->ddir == DDIR_SYNC || io_u->ddir == DDIR_DATASYNC)
		return FIO_Q_COMPLETED;

	if (!ddir_rw(io_u->ddir)) {
		io_u->error = EOPNOTSUPP;
		td_verror(td, EOPNOTSUPP, "rocm-xio unsupported ddir");
		return FIO_Q_COMPLETED;
	}

	fio_ro_check(td, io_u);
	if (io_u->error)
		return FIO_Q_COMPLETED;

	if ((io_u->offset % xd->lba_size) || (io_u->xfer_buflen % xd->lba_size)) {
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "rocm-xio unaligned IO");
		return FIO_Q_COMPLETED;
	}

	lbas = io_u->xfer_buflen / xd->lba_size;
	if (!lbas || lbas > UINT32_MAX) {
		io_u->error = EINVAL;
		td_verror(td, EINVAL, "rocm-xio invalid transfer size");
		return FIO_Q_COMPLETED;
	}

	xd->endpoint_cfg->ioParams.baseLba = io_u->offset / xd->lba_size;
	xd->endpoint_cfg->ioParams.lbasPerIo = static_cast<uint32_t>(lbas);
	xd->endpoint_cfg->ioParams.readIo = (io_u->ddir == DDIR_READ) ? 1 : 0;
	xd->endpoint_cfg->ioParams.writeIo = (io_u->ddir == DDIR_WRITE) ? 1 : 0;
	xd->endpoint_cfg->bufferParams.bufferSize = io_u->xfer_buflen;

	xd->run_cfg.iterations = 1;
	xd->run_cfg.verifyPass = 0;
	xd->run_cfg.verifyFail = 0;

	run_err = xd->endpoint->run(&xd->run_cfg);
	if (run_err != hipSuccess) {
		log_err("rocm-xio: run failed: %s (%d)\n",
			hipGetErrorString(run_err), run_err);
		io_u->error = EIO;
		td_verror(td, EIO, "rocm-xio run");
	}

	return FIO_Q_COMPLETED;
}

static void fio_rocm_xio_cleanup(struct thread_data *td)
{
	auto *xd = static_cast<rocm_xio_data *>(td->io_ops_data);

	delete xd;
	td->io_ops_data = nullptr;
}

static int fio_rocm_xio_open_file(struct thread_data fio_unused *td,
				  struct fio_file fio_unused *f)
{
	return 0;
}

static int fio_rocm_xio_close_file(struct thread_data fio_unused *td,
				   struct fio_file fio_unused *f)
{
	return 0;
}

FIO_STATIC struct ioengine_ops ioengine = {
	.name			= "rocm-xio",
	.version		= FIO_IOOPS_VERSION,
	.init			= fio_rocm_xio_init,
	.queue			= fio_rocm_xio_queue,
	.open_file		= fio_rocm_xio_open_file,
	.close_file		= fio_rocm_xio_close_file,
	.cleanup		= fio_rocm_xio_cleanup,
	.flags			= FIO_SYNCIO | FIO_DISKLESSIO | FIO_NOEXTEND |
				  FIO_NODISKUTIL,
	.options		= options,
	.option_struct_size	= sizeof(struct rocm_xio_options),
};

extern "C" void fio_init fio_rocm_xio_register(void)
{
	register_ioengine(&ioengine);
}

extern "C" void fio_exit fio_rocm_xio_unregister(void)
{
	unregister_ioengine(&ioengine);
}
