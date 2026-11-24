# SPDX-License-Identifier: Apache-2.0
"""Process isolation for KDA tests using the installed MetaX native operator."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest


def _init_metax_worker(device_index):
    # Do not import FlagGems at module scope: spawn imports this module before
    # running the initializer. mcoplib must register its schemas first.
    import torch

    from baseline.kda_recoverssm_verify import native

    torch.cuda.set_device(device_index)
    native()


@pytest.fixture(scope="module")
def kda_worker():
    import flaggems_vllm as gems

    if gems.vendor_name != "metax":
        yield None
        return
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_metax_worker,
        initargs=(gems.runtime.torch_device_fn.current_device(),),
    ) as pool:
        yield pool
