# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Native vLLM KDA RecoverSSM speculative-verify workloads and signatures.

The native callable is ``kda_recoverssm_verify`` from
``vllm.models.kimi_k3.nvidia.ops.recoverssm`` -- a pure Triton kernel. Some
installed vLLM builds cannot load their compiled extensions (``vllm._C`` /
``vllm.vllm_flash_attn``, e.g. CUDA 13 extensions installed on a cu128 torch).
Importing the module through the package path then dies inside
``vllm.models.kimi_k3``'s eager package ``__init__`` long before the
Triton-only ops submodule is reached. ``native()`` then falls back to
registering stub parent packages so the submodule import skips that
``__init__``; on healthy installs the plain import succeeds and nothing is
patched. The kernel itself never touches the compiled extensions.
"""

import math
import os
import sys
import types
from importlib import import_module
from typing import NamedTuple, Optional

import torch

OP_NAME = "kda_recoverssm_verify"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ("out", "records")
NATIVE_UNAVAILABLE_REASON = None
INPUT_SEED = 20261118

_MODULE = None
_RECOVERSSM = "vllm.models.kimi_k3.nvidia.ops.recoverssm"


def _recoverssm_module():
    """Import vLLM's RecoverSSM ops module, bypassing a broken package init."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    try:
        _MODULE = import_module(_RECOVERSSM)
        return _MODULE
    except ImportError as first_error:
        import vllm

        root = os.path.dirname(os.path.abspath(vllm.__file__))
        for pkg in ("vllm.models.kimi_k3", "vllm.models.kimi_k3.nvidia"):
            # A failed first attempt can leave half-initialized parents in
            # sys.modules; replace any of them with a stub and retry once.
            # If the parents were healthy and the submodule itself is broken,
            # the retry fails at the same place and surfaces that error.
            sys.modules.pop(pkg, None)
            stub = types.ModuleType(pkg)
            stub.__path__ = [os.path.join(root, *pkg.split(".")[1:])]
            stub.__package__ = pkg
            sys.modules[pkg] = stub
        try:
            _MODULE = import_module(_RECOVERSSM)
        except ImportError:
            raise first_error
        return _MODULE


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        fn = _recoverssm_module().kda_recoverssm_verify
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return fn


class Case(NamedTuple):
    name: str
    lengths: list  # speculative tokens per request, each <= window
    window: int = 2  # spec_query_len = 1 + num_spec (K3 MTP window: 2..8)
    heads: int = 12  # per-rank KDA heads (K3: 96 global -> 96/24/48/12 at TP1/2/4/8)
    head_dim: int = 128  # KDA: keys = values = head_dim (K3 config: 128)
    lower_bound: Optional[float] = -5.0  # K3 gate_lower_bound; None = softplus path
    null_requests: tuple = ()  # requests without a checkpoint (NULL_BLOCK_ID)


PERF_CASES = [
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
    # Edge shapes: ragged windows, missing checkpoints, unbounded gate.
    Case("ragged", [8, 3, 6, 1, 8], window=8, heads=12),
    Case("null_ckpt", [4] * 4, window=4, heads=12, null_requests=(1, 3)),
    Case("lb_none_small", [2] * 4, window=2, heads=4, lower_bound=None),
]


def grid():
    return [{"case": case} for case in PERF_CASES]


def key_shape(binding):
    case = Case(*binding["case"])
    lb = "none" if case.lower_bound is None else f"{case.lower_bound:g}"
    return (
        f"req{len(case.lengths)}x{case.window}_h{case.heads}_d{case.head_dim}"
        f"_lb{lb}"
    )


def build(case, dtype, device):
    case = Case(*case)
    gen = torch.Generator().manual_seed(INPUT_SEED)
    module = _recoverssm_module()
    try:
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    except ImportError:
        NULL_BLOCK_ID = 0  # vllm.v1.attention.backends.utils.NULL_BLOCK_ID
    n, total = len(case.lengths), sum(case.lengths)
    h, d = case.heads, case.head_dim

    def rand(shape, dt=dtype, scale=0.2):
        return (torch.randn(shape, generator=gen) * scale).to(dt).to(device)

    q = rand((1, total, h, d))
    k = rand((1, total, h, d))
    v = rand((1, total, h, d))
    raw_g = rand((1, total, h, d))
    raw_beta = rand((1, total, h))
    a_log = rand((h,), torch.float32)
    dt_bias = rand((h * d,), torch.float32)
    num_blocks = n + 4
    checkpoint_state = rand((num_blocks, h, d, d))
    records = module.KDARecoverSSMRecords.allocate(
        n, h, case.window, d, dtype, torch.device(device)
    )
    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(case.lengths).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    free = iter(range(NULL_BLOCK_ID + 1, num_blocks))
    state_indices = torch.tensor(
        [NULL_BLOCK_ID if b in case.null_requests else next(free) for b in range(n)],
        dtype=torch.int32,
        device=device,
    )
    return dict(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=case.lower_bound,
        checkpoint_state=checkpoint_state,
        records=records,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=case.window,
    )


def build_inputs(binding, dtype, device):
    return (), build(Case(*binding["case"]), dtype, device)


def unsupported_reason(binding, dtype):
    return None


def config(binding, dtype):
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
        "checkpoint_state": tensor([num_blocks, h, d, d], dtype),
        "records": {
            "correction": tensor([n, h, w, d], torch.float32),
            "key_gate": tensor([n, h, w, 2 * d], dtype),
        },
        "query_start_loc": tensor([n + 1], torch.int32),
        "state_indices": tensor([n], torch.int32),
        "spec_query_len": {"scalar": w},
    }
    return dict(
        inputs=inputs,
        outputs={
            "out": tensor([1, total, h, d], dtype),
            "records.correction": tensor([n, h, w, d], torch.float32),
            "records.key_gate": tensor([n, h, w, 2 * d], dtype),
        },
        dims=case._asdict(),
        accumulate_dtype="torch.float32",
        input_seed=INPUT_SEED,
        mutation=["out", "records (verify records rewritten every call)"],
        native_unsupported_reason=unsupported_reason(binding, dtype),
        graph=False,
    )
