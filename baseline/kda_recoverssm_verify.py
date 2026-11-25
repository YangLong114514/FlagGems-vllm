# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Native vLLM KDA RecoverSSM speculative-verify workloads and signatures.

Hygon vLLM 0.28.1+dtk2604 and MetaX 0.29 ship the same ``recoverssm.py``
as upstream v0.28.1rc0 and v0.29.0. Their verify kernel and the Moore Threads
0.32 verify kernel write ``correction_cache`` and ``kg_cache`` by physical
checkpoint block. Upstream v0.33.0 and v0.34.0 write ``records`` by speculative
batch row. Tests and benchmarks select the layout from the installed signature.
"""

import inspect
import math
from contextlib import nullcontext
from functools import lru_cache
from types import SimpleNamespace
from typing import NamedTuple, Optional
from unittest.mock import patch

import torch

OP_NAME = "kda_recoverssm_verify"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
INPUT_SEED = 20261118
NULL_BLOCK_ID = 0  # vLLM's reserved block for requests without a checkpoint.


@lru_cache(maxsize=1)
def native():
    from vllm.platforms import current_platform

    capability_probe = nullcontext()
    if hasattr(type(current_platform), "is_ppu") and current_platform.is_ppu():
        from vllm.platforms.cuda import NonNvmlCudaPlatform, pynvml

        # Importing kimi_k3 probes CUTLASS via PPUPlatform's inherited NVML
        # method. SAIL PyTorch can query the same device without that library.
        try:
            current_platform.get_device_capability()
        except pynvml.NVMLError_LibraryNotFound:
            capability_probe = patch.object(
                type(current_platform),
                "get_device_capability",
                NonNvmlCudaPlatform.get_device_capability,
            )

    with capability_probe:
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import kda_recoverssm_verify

    return kda_recoverssm_verify


def _uses_block_cache():
    params = inspect.signature(native()).parameters
    if "records" in params:
        return False
    if {"correction_cache", "kg_cache"} <= params.keys():
        return True
    raise ValueError(f"Unsupported native KDA verify signature: {list(params)}")


def _record_kwargs(inputs, *, for_native):
    k, v = inputs["k"], inputs["v"]
    heads, key_dim = k.shape[2:]
    batch = inputs["state_indices"].numel()
    window = inputs["spec_query_len"]
    if for_native and _uses_block_cache():
        rows = inputs["checkpoint_state"].shape[0]
        shape = (rows, heads, window)
        return dict(
            correction_cache=v.new_empty((*shape, v.shape[-1]), dtype=torch.float32),
            kg_cache=k.new_empty((*shape, 2 * key_dim)),
        )
    if for_native:
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMRecords

        records = KDARecoverSSMRecords.allocate(
            batch, heads, window, key_dim, k.dtype, k.device
        )
    else:
        shape = (batch, heads, window)
        records = SimpleNamespace(
            correction=v.new_empty((*shape, v.shape[-1]), dtype=torch.float32),
            key_gate=k.new_empty((*shape, 2 * key_dim)),
        )
    return dict(records=records)


def native_inputs(inputs):
    """Share read-only workload tensors and allocate independent native records."""
    kwargs = inputs.copy()
    kwargs.pop("records")
    kwargs.pop("out", None)
    kwargs.update(_record_kwargs(kwargs, for_native=True))
    return kwargs


class Case(NamedTuple):
    name: str
    lengths: list  # speculative tokens per request, each <= window
    window: int = 2  # spec_query_len = 1 + num_spec (K3 MTP window: 2..8)
    heads: int = 12  # per-rank KDA heads (K3: 96 global -> 96/24/48/12 at TP1/2/4/8)
    head_dim: int = 128  # KDA: keys = values = head_dim (K3 config: 128)
    lower_bound: Optional[float] = -5.0  # K3 gate_lower_bound; None = softplus path
    null_requests: tuple = ()  # requests without a checkpoint (NULL_BLOCK_ID)


PRODUCTION_CASES = [
    # Kimi K3 production shapes: window = 1 + num_spec (MTP-1 -> 2, spec-3 -> 4,
    # max 8), per-rank heads = 96/TP, decode batch up to 256, head_dim 128,
    # gate_lower_bound -5.0.
    Case("k3_tp8_mtp1_bs32", [2] * 32, window=2, heads=12),
    Case("k3_tp8_mtp1_bs128", [2] * 128, window=2, heads=12),
    Case("k3_tp8_mtp1_bs256", [2] * 256, window=2, heads=12),
    Case("k3_tp8_spec3_bs32", [4] * 32, window=4, heads=12),
    Case("k3_tp8_spec3_bs128", [4] * 128, window=4, heads=12),
    Case("k3_tp8_win8_bs32", [8] * 32, window=8, heads=12),
    Case("k3_tp4_mtp1_bs64", [2] * 64, window=2, heads=24),
    Case("k3_tp2_mtp1_bs64", [2] * 64, window=2, heads=48),
    Case("k3_tp1_mtp1_bs8", [2] * 8, window=2, heads=96),
    Case("k3_tp1_spec3_bs16", [4] * 16, window=4, heads=96),
]

EDGE_CASES = [
    # Functional edge cases, excluded from production performance aggregation.
    Case("ragged", [8, 3, 6, 1, 8], window=8, heads=12),
    Case("null_ckpt", [4] * 4, window=4, heads=12, null_requests=(1, 3)),
    Case("lb_none_small", [2] * 4, window=2, heads=4, lower_bound=None),
]

PERF_CASES = PRODUCTION_CASES + EDGE_CASES


def grid():
    return [{"case": case} for case in PERF_CASES]


def key_shape(binding):
    case = Case(*binding["case"])
    lb = "none" if case.lower_bound is None else f"{case.lower_bound:g}"
    return (
        f"req{len(case.lengths)}x{case.window}_h{case.heads}_d{case.head_dim}"
        f"_lb{lb}"
    )


def build(
    case, dtype, device, state_dtype=torch.float32, *, vendor=None, for_native=False
):
    case = Case(*case)
    gen = torch.Generator().manual_seed(INPUT_SEED)
    n, total = len(case.lengths), sum(case.lengths)
    h, d = case.heads, case.head_dim

    def rand(shape, dt=dtype, scale=0.2):
        if vendor in ("metax", "mthreads"):
            # Form [1,T,H,D] after transfer so the head/dim strides are (D,1).
            flat = torch.randn(math.prod(shape), generator=gen)
            return (flat * scale).to(dt).to(device).view(shape)
        return (torch.randn(shape, generator=gen) * scale).to(dt).to(device)

    q = rand((1, total, h, d))
    k = rand((1, total, h, d))
    v = rand((1, total, h, d))
    raw_g = rand((1, total, h, d))
    raw_beta = rand((1, total, h))
    a_log = rand((h,), torch.float32)
    dt_bias = rand((h * d,), torch.float32)
    num_blocks = n + 4
    checkpoint_state = rand((num_blocks, h, d, d), state_dtype)
    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(case.lengths).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    blocks = range(NULL_BLOCK_ID + 1, num_blocks)
    # Cover non-sequential physical slots in the NULL-checkpoint edge case.
    free = reversed(blocks) if case.null_requests else iter(blocks)
    state_indices = torch.tensor(
        [NULL_BLOCK_ID if b in case.null_requests else next(free) for b in range(n)],
        dtype=torch.int32,
        device=device,
    )
    inputs = dict(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=case.lower_bound,
        checkpoint_state=checkpoint_state,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=case.window,
    )
    inputs.update(_record_kwargs(inputs, for_native=for_native))
    return inputs


def build_inputs(binding, dtype, device):
    return (), build(Case(*binding["case"]), dtype, device, for_native=True)


def unsupported_reason(binding, dtype):
    return None


def config(binding, dtype, state_dtype=torch.float32):
    case = Case(*binding["case"])
    n, total = len(case.lengths), sum(case.lengths)
    h, d, w = case.heads, case.head_dim, case.window
    num_blocks = n + 4

    def tensor(shape, dt):
        return dict(
            shape=shape,
            dtype=str(dt),
            stride=[math.prod(shape[i + 1 :]) for i in range(len(shape))],
        )

    inputs = {
        "q": tensor([1, total, h, d], dtype),
        "k": tensor([1, total, h, d], dtype),
        "v": tensor([1, total, h, d], dtype),
        "raw_g": tensor([1, total, h, d], dtype),
        "raw_beta": tensor([1, total, h], dtype),
        "A_log": tensor([h], torch.float32),
        "dt_bias": tensor([h * d], torch.float32),
        "lower_bound": {"scalar": case.lower_bound},
        "checkpoint_state": tensor([num_blocks, h, d, d], state_dtype),
        "query_start_loc": tensor([n + 1], torch.int32),
        "state_indices": tensor([n], torch.int32),
        "spec_query_len": {"scalar": w},
    }
    outputs = {"out": tensor([1, total, h, d], dtype)}
    if _uses_block_cache():
        inputs["correction_cache"] = tensor([num_blocks, h, w, d], torch.float32)
        inputs["kg_cache"] = tensor([num_blocks, h, w, 2 * d], dtype)
        outputs.update({key: inputs[key] for key in ("correction_cache", "kg_cache")})
    else:
        inputs["records"] = {
            "correction": tensor([n, h, w, d], torch.float32),
            "key_gate": tensor([n, h, w, 2 * d], dtype),
        }
        outputs.update(
            {f"records.{key}": val for key, val in inputs["records"].items()}
        )
    return dict(
        inputs=inputs,
        outputs=outputs,
        dims=case._asdict(),
        accumulate_dtype="torch.float32",
        input_seed=INPUT_SEED,
        mutation=list(outputs),
        native_unsupported_reason=unsupported_reason(binding, dtype),
        graph=False,
    )
