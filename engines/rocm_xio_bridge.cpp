/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "rocm_xio_bridge.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <string>

#include <hip/hip_runtime.h>
#include <xio.h>
#include "endpoints/nvme-ep/nvme-ep.h"

static int set_error(char *errbuf, size_t errbuf_len, const char *fmt, ...)
{
	va_list args;

	if (!errbuf || !errbuf_len)
		return -1;

	va_start(args, fmt);
	vsnprintf(errbuf, errbuf_len, fmt, args);
	va_end(args);

	return -1;
}

static int normalize_ret(int ret)
{
	if (ret == 0)
		return 0;
	if (ret < 0)
		return ret;
	return ret;
}

extern "C" int rocm_xio_query_lba_size(const char *controller, uint32_t nsid,
				       uint32_t *lba_size, char *errbuf,
				       size_t errbuf_len)
{
	unsigned detected = 0;
	int ret;

	if (!controller || !controller[0] || !lba_size)
		return set_error(errbuf, errbuf_len, "invalid lba size query arguments");

	ret = xio::nvme_ep::queryLbaSize(controller, nsid, &detected);
	ret = normalize_ret(ret);
	if (ret)
		return set_error(errbuf, errbuf_len,
				 "xio::nvme_ep::queryLbaSize(%s, nsid=%u) failed (%d)",
				 controller, nsid, ret);

	*lba_size = detected;
	if (errbuf && errbuf_len)
		errbuf[0] = '\0';
	return 0;
}

extern "C" int rocm_xio_query_namespace_capacity(const char *controller,
						 uint32_t nsid,
						 uint64_t *capacity_lbas,
						 char *errbuf,
						 size_t errbuf_len)
{
	uint64_t detected = 0;
	int ret;

	if (!controller || !controller[0] || !capacity_lbas)
		return set_error(errbuf, errbuf_len,
				 "invalid namespace capacity query arguments");

	ret = xio::nvme_ep::queryNamespaceCapacity(controller, nsid, &detected);
	ret = normalize_ret(ret);
	if (ret)
		return set_error(errbuf, errbuf_len,
				 "xio::nvme_ep::queryNamespaceCapacity(%s, nsid=%u) failed (%d)",
				 controller, nsid, ret);

	*capacity_lbas = detected;
	if (errbuf && errbuf_len)
		errbuf[0] = '\0';
	return 0;
}

extern "C" int rocm_xio_submit(const struct rocm_xio_submit_opts *opts,
			       uint32_t *verify_pass, uint32_t *verify_fail,
			       char *errbuf, size_t errbuf_len)
{
	xio::XioEndpointConfig config = {};
	xio::nvme_ep::nvmeEpConfig nvme = {};
	std::string validation;
	hipError_t hret;
	uint32_t lba_size = 0;
	uint16_t queue_id;
	uint16_t queue_len;
	uint16_t num_queues;
	size_t xfer_bytes;
	int ret;

	if (!opts || !opts->controller || !opts->controller[0])
		return set_error(errbuf, errbuf_len, "invalid rocm-xio submit options");

	if (!opts->do_read && !opts->do_write)
		return set_error(errbuf, errbuf_len,
				 "rocm-xio submit requires do_read or do_write");

	if (!opts->lbas_per_io)
		return set_error(errbuf, errbuf_len,
				 "rocm-xio submit requires lbas_per_io > 0");

	if (opts->queue_id > UINT16_MAX || opts->queue_length > UINT16_MAX ||
	    opts->num_queues > UINT16_MAX)
		return set_error(errbuf, errbuf_len,
				 "rocm-xio queue parameters exceed uint16 limits");

	queue_id = (uint16_t)opts->queue_id;
	queue_len = (uint16_t)(opts->queue_length ? opts->queue_length : 64);
	num_queues = (uint16_t)(opts->num_queues ? opts->num_queues : 1);

	ret = rocm_xio_query_lba_size(opts->controller, opts->nsid ? opts->nsid : 1,
				      &lba_size, errbuf, errbuf_len);
	if (ret)
		return ret;

	xfer_bytes = (size_t)opts->lbas_per_io * lba_size;

	config.numThreads = 1;
	config.delayNs = 0;
	config.memoryMode = opts->memory_mode;
	config.verbose = false;
	config.pciMmioBridge = opts->use_pci_mmio_bridge ? true : false;
	config.startTimes = nullptr;
	config.endTimes = nullptr;
	config.timingStats = nullptr;
	config.substepStats = nullptr;
	config.stopRequested = nullptr;
	config.verifyPass = 0;
	config.verifyFail = 0;

	nvme.controller.assign(opts->controller);
	nvme.queueId = queue_id;
	nvme.queueLength = queue_len;
	nvme.numQueues = num_queues;
	nvme.verify = opts->verify ? true : false;

	/*
	 * fio already decides offsets and randomization. Submit a single
	 * deterministic operation at the exact LBA requested by fio.
	 */
	nvme.ioParams.accessPattern = "sequential";
	nvme.ioParams.readIo = opts->do_read ? 1 : 0;
	nvme.ioParams.writeIo = opts->do_write ? 1 : 0;
	nvme.ioParams.nsid = opts->nsid ? opts->nsid : 1;
	nvme.ioParams.baseLba = opts->base_lba;
	nvme.ioParams.lbasPerIo = opts->lbas_per_io;
	nvme.ioParams.lfsrSeed = opts->lfsr_seed;
	nvme.ioParams.batchSize = opts->batch_size ? opts->batch_size : 1;
	nvme.ioParams.infiniteMode = false;

	nvme.bufferParams.bufferSize = std::max(xfer_bytes, (size_t)4096);
	nvme.doorbellParams.usePciMmioBridge = opts->use_pci_mmio_bridge ? true : false;

	validation = xio::nvme_ep::validateConfig(&nvme);
	if (!validation.empty())
		return set_error(errbuf, errbuf_len, "xio validateConfig failed: %s",
				 validation.c_str());

	config.endpointConfig = &nvme;

	hret = xio::nvme_ep::run(&config);
	if (hret != hipSuccess)
		return set_error(errbuf, errbuf_len, "xio run failed (hip=%d)",
				 (int)hret);

	if (verify_pass)
		*verify_pass = config.verifyPass;
	if (verify_fail)
		*verify_fail = config.verifyFail;

	if (errbuf && errbuf_len)
		errbuf[0] = '\0';
	return 0;
}
