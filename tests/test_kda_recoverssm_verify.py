# SPDX-License-Identifier: Apache-2.0
"""KDA RecoverSSM speculative verify: native baseline and Gems parity.

The native vLLM baseline is resolved through
``baseline/kda_recoverssm_verify.py`` and checked against an independent fp64
torch oracle (allowed in tests only); the FlagGems-vllm implementation is
then checked against the native outputs on Kimi K3 production shapes and
against the interface contract of the vLLM kernel.
"""

import types

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm as gems
from baseline.kda_recoverssm_verify import PERF_CASES, Case, build, native

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.kda_recoverssm_verify,
]

NATIVE = native()
DTYPES = [torch.float16, torch.bfloat16]
PRECISIONS = [
    pytest.param(torch.bfloat16, torch.float32, id="bf16_activation_fp32_state"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="bf16_activation_bf16_state"),
    pytest.param(torch.float16, torch.float32, id="fp16_activation_fp32_state"),
]
# bf16/fp16 outputs are produced by fp32 math rounded once on store.
OUT_TOL = {torch.float16: 1.5e-2, torch.bfloat16: 3e-2}
CORRECTION_TOL = 1e-3  # records.correction is stored in fp32


def reference(case, inputs):
    """Vectorized fp64 torch oracle of the kernel semantics (test-only).

    Returns (out [1,T,H,V] fp64, corrections [B,H,W,V] fp64, written [B,W]
    bool). Slots with written=False (padding of short windows and NULL
    requests) are not stored by the kernel and must not be compared.
    """
    case = Case(*case)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    raw_g, raw_beta = inputs["raw_g"], inputs["raw_beta"]
    batch, heads, dim = len(case.lengths), case.heads, case.head_dim
    window = inputs["spec_query_len"]
    total_tokens = q.shape[1]
    bos = torch.tensor(inputs["query_start_loc"][:-1].tolist(), device=q.device)
    lengths = torch.tensor(case.lengths, device=q.device)
    state_idx = inputs["state_indices"]
    has_state = state_idx > 0  # NULL_BLOCK_ID: the kernel writes zeros
    state = inputs["checkpoint_state"][state_idx.clamp(min=0)]
    state = state.to(torch.float64).clone()  # [B,H,V,K]
    dt_bias = inputs["dt_bias"].to(torch.float64).view(heads, dim)
    a = torch.exp(inputs["A_log"].to(torch.float64))  # [H]
    out = torch.zeros_like(v, dtype=torch.float64)
    corr_all = torch.zeros(
        batch, heads, window, v.shape[-1], dtype=torch.float64, device=q.device
    )
    written = torch.zeros(batch, window, dtype=torch.bool, device=q.device)
    for off in range(max(case.lengths, default=0)):
        t = (bos + off).clamp(max=total_tokens - 1)
        valid = (off < lengths) & has_state
        qn = q[0, t].to(torch.float64)  # [B,H,K]
        kn = k[0, t].to(torch.float64)
        gate_in = raw_g[0, t].to(torch.float64) + dt_bias
        if case.lower_bound is not None:
            gate = case.lower_bound * torch.sigmoid(a[None, :, None] * gate_in)
        else:
            gate = -a[None, :, None] * F.softplus(gate_in, beta=1.0, threshold=20.0)
        qn = qn * torch.rsqrt((qn * qn).sum(-1, keepdim=True) + 1e-6) * (dim**-0.5)
        kn = kn * torch.rsqrt((kn * kn).sum(-1, keepdim=True) + 1e-6)
        state = state * torch.exp(gate)[:, :, None, :]
        corr = v[0, t].to(torch.float64) - (state * kn[:, :, None, :]).sum(-1)
        corr = corr * torch.sigmoid(raw_beta[0, t].to(torch.float64))[..., None]
        new_state = state + corr[..., None] * kn[:, :, None, :]
        state = torch.where(valid[:, None, None, None], new_state, state)
        step_out = (state * qn[:, :, None, :]).sum(-1)  # [B,H,V]
        out[0, t] = torch.where(valid[:, None, None], step_out, out[0, t])
        corr_all[:, :, off] = corr
        written[:, off] = valid
    return out, corr_all, written


def assert_records_match(case, inputs, records, ref_corr, written):
    case = Case(*case)
    d = case.head_dim
    for b in range(len(case.lengths)):
        for off in range(case.lengths[b]):
            if not written[b, off]:
                continue
            rec = records.correction[b, :, off].to(torch.float64)
            err = (rec - ref_corr[b, :, off]).abs().max().item()
            assert err <= CORRECTION_TOL, f"correction err {err:.3e} at ({b}, {off})"
            t = int(inputs["query_start_loc"][b]) + off
            assert torch.equal(records.key_gate[b, :, off, :d], inputs["k"][0, t])
            assert torch.equal(records.key_gate[b, :, off, d:], inputs["raw_g"][0, t])


@pytest.mark.parametrize("case", PERF_CASES, ids=[c.name for c in PERF_CASES])
@pytest.mark.parametrize("dtype,state_dtype", PRECISIONS)
def test_native_against_torch_oracle(case, dtype, state_dtype):
    inputs = build(case, dtype, "cuda", state_dtype)
    checkpoint_before = inputs["checkpoint_state"].clone()

    out = NATIVE(**inputs)
    torch.cuda.synchronize()

    ref_out, ref_corr, written = reference(case, inputs)
    case = Case(*case)

    err = (out.to(torch.float64) - ref_out).abs().max().item()
    assert err <= OUT_TOL[dtype], f"out max abs err {err:.3e}"
    for b in case.null_requests:
        bos = int(inputs["query_start_loc"][b])
        eos = int(inputs["query_start_loc"][b + 1])
        assert out[0, bos:eos].abs().max().item() == 0.0
    assert torch.equal(inputs["checkpoint_state"], checkpoint_before)
    assert_records_match(case, inputs, inputs["records"], ref_corr, written)


@pytest.mark.parametrize("case", PERF_CASES, ids=[c.name for c in PERF_CASES])
@pytest.mark.parametrize("dtype,state_dtype", PRECISIONS)
def test_gems_matches_native(case, dtype, state_dtype):
    native_inputs = build(case, dtype, "cuda", state_dtype)
    gems_inputs = build(case, dtype, "cuda", state_dtype)
    checkpoint_before = gems_inputs["checkpoint_state"].clone()

    native_out = NATIVE(**native_inputs)
    gems_out = gems.kda_recoverssm_verify(**gems_inputs)
    torch.cuda.synchronize()

    # Every out row is a valid token of some request, so out is fully
    # written; only reduction-order ULPs may differ between configs.
    rtol, atol = {torch.float16: (3e-3, 3e-3), torch.bfloat16: (2e-2, 2e-2)}[dtype]
    torch.testing.assert_close(gems_out, native_out, rtol=rtol, atol=atol)
    # Record slots beyond a request's token count (and rows of NULL
    # requests) are never written by either kernel; compare written slots.
    case = Case(*case)
    for b, length in enumerate(case.lengths):
        if b in case.null_requests:
            continue
        native_corr = native_inputs["records"].correction[b, :, :length]
        gems_corr = gems_inputs["records"].correction[b, :, :length]
        torch.testing.assert_close(gems_corr, native_corr, rtol=1e-4, atol=1e-5)
        native_kg = native_inputs["records"].key_gate[b, :, :length]
        gems_kg = gems_inputs["records"].key_gate[b, :, :length]
        assert torch.equal(gems_kg, native_kg)
    assert torch.equal(gems_inputs["checkpoint_state"], checkpoint_before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_out_argument_and_duck_typed_records(dtype):
    case = Case("k3_tp8_mtp1_bs32", [2] * 32, window=2, heads=12)
    inputs = build(case, dtype, "cuda")
    inputs["records"] = types.SimpleNamespace(
        correction=torch.empty_like(inputs["records"].correction),
        key_gate=torch.empty_like(inputs["records"].key_gate),
    )
    out = torch.empty_like(inputs["v"])
    returned = gems.kda_recoverssm_verify(**inputs, out=out)
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    ref_out, ref_corr, written = reference(case, inputs)
    err = (out.to(torch.float64) - ref_out).abs().max().item()
    assert err <= OUT_TOL[dtype], f"out max abs err {err:.3e}"
    assert_records_match(case, inputs, inputs["records"], ref_corr, written)


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_tokens_early_return(dtype):
    case = Case("k3_tp8_mtp1_bs32", [2] * 32, window=2, heads=12)
    inputs = build(case, dtype, "cuda")
    for name in ("q", "k", "v", "raw_g"):
        inputs[name] = inputs[name][:, :0]
    inputs["raw_beta"] = inputs["raw_beta"][:, :0]
    inputs["query_start_loc"] = torch.zeros_like(inputs["query_start_loc"])
    out = gems.kda_recoverssm_verify(**inputs)
    assert out.shape == inputs["v"].shape


CONTRACT_ERRORS = [
    # (mutator, message fragment)
    (
        lambda i: i.update(q=i["q"].squeeze(0)),
        "q must have shape",
    ),
    (
        lambda i: i.update(k=i["k"][:, :, :, : i["k"].shape[-1] // 2]),
        "q, k, and v shapes are incompatible",
    ),
    (
        lambda i: i.update(raw_beta=i["raw_beta"][..., :1]),
        "gate or beta shape is incompatible",
    ),
    (
        lambda i: i.update(
            q=torch.empty(
                (1, i["q"].shape[1], i["q"].shape[2], i["q"].shape[3] * 2),
                dtype=i["q"].dtype,
                device="cuda",
            )[..., : i["q"].shape[3]].copy_(i["q"])
        ),
        "heads must be contiguous",
    ),
    (
        lambda i: i.update(checkpoint_state=i["checkpoint_state"][:, :, :, :-1]),
        "checkpoint shape is incompatible",
    ),
    (
        lambda i: i.update(A_log=i["A_log"][:-1]),
        "gate parameters are incompatible",
    ),
    (
        lambda i: i.update(dt_bias=i["dt_bias"][::2]),
        "gate parameters",
    ),
    (
        lambda i: i.update(query_start_loc=i["query_start_loc"][:-1]),
        "query metadata is incompatible",
    ),
    (
        lambda i: i.update(out=torch.empty(1, 1, 1, 1, dtype=i["v"].dtype, device="cuda")),
        "output shape is incompatible",
    ),
]


@pytest.mark.parametrize("mutate, message", CONTRACT_ERRORS)
def test_contract_errors(mutate, message):
    case = Case("contract", [2] * 4, window=2, heads=4)
    inputs = build(case, torch.bfloat16, "cuda")
    mutate(inputs)
    with pytest.raises(ValueError, match=message):
        gems.kda_recoverssm_verify(**inputs)


def test_device_mismatch_rejected():
    case = Case("contract", [2] * 4, window=2, heads=4)
    inputs = build(case, torch.bfloat16, "cuda")
    inputs["state_indices"] = inputs["state_indices"].cpu()
    with pytest.raises(ValueError, match="same device"):
        gems.kda_recoverssm_verify(**inputs)
