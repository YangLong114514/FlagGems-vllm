# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend implementation of KDA RecoverSSM speculative verify.

The common recurrence is executed by a persistent Vector-core kernel. Each
program walks multiple work tiles, avoiding the global auto-blockify switch and
its cross-operator correctness risks.
"""

from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._ascend.device import get_device_core_counts
from flaggems_vllm.utils import libentry, libtuner

# Matches vllm.v1.attention.backends.utils.NULL_BLOCK_ID; requests whose
# checkpoint slot is <= null_block_id have no state and produce zero outputs.
DEFAULT_NULL_BLOCK_ID = 0


@lru_cache(maxsize=None)
def _vector_core_count(device: torch.device) -> int:
    return get_device_core_counts(device)[1]


@triton.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        return lower_bound * tl.sigmoid(A * gate_input)
    softplus_gate = tl.where(
        gate_input > 20.0,
        gate_input,
        tl.log(1.0 + tl.exp(gate_input)),
    )
    return -A * softplus_gate


@triton.jit
def _kda_recurrent_step(
    state,
    k,
    v,
    raw_g,
    raw_beta,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    gate = _kda_gate(
        raw_g,
        dt_bias,
        A,
        lower_bound,
        USE_LOWER_BOUND,
    )

    state *= tl.exp(gate)[None, :]
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
    correction *= tl.sigmoid(raw_beta)
    return state + correction[:, None] * normalized_k[None, :], correction


@triton.jit
def _kda_prepared_step(state, normalized_k, q, v, decay, beta):
    state *= decay[None, :]
    state_q = tl.sum(state * q[None, :], axis=1)
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
    correction *= beta
    return state + correction[:, None] * normalized_k[None, :], correction, state_q


@triton.jit
def _ascend_kda_recoverssm_verify_pair_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_row,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_row,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    work_id,
    NUM_HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid_b = work_id // NUM_HEADS
    pid_h = work_id % NUM_HEADS
    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, BV)
    if state_idx <= null_block_id:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            for pid_v in tl.static_range(V // BV):
                tl.store(
                    out_ptr
                    + (bos + token_offset) * stride_out_token
                    + pid_h * V
                    + pid_v * BV
                    + offs_v,
                    tl.zeros([BV], dtype=tl.float32),
                    mask=token_valid,
                )
        query_len = tl.zeros((), dtype=tl.int64)
    state_idx = tl.maximum(state_idx, 0)
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k).to(tl.float32)

    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        token = bos + token_offset
        token_valid = token_offset < query_len
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q) + 1e-6) * (K**-0.5)
        normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        normalized_k_q = tl.sum(normalized_k * q)
        gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
        tl.store(
            kg_cache_ptr
            + pid_b * stride_kg_row
            + pid_h * stride_kg_head
            + token_offset * stride_kg_pos
            + offs_k * stride_kg_dim,
            k,
            mask=token_valid,
        )
        tl.store(
            kg_cache_ptr
            + pid_b * stride_kg_row
            + pid_h * stride_kg_head
            + token_offset * stride_kg_pos
            + (K + offs_k) * stride_kg_dim,
            raw_g,
            mask=token_valid,
        )
        if token_offset == 0:
            q0 = q
            normalized_k0 = normalized_k
            normalized_k_q0 = normalized_k_q
            decay0 = tl.exp(gate)
            beta0 = tl.sigmoid(raw_beta)
        elif token_offset == 1:
            q1 = q
            normalized_k1 = normalized_k
            normalized_k_q1 = normalized_k_q
            decay1 = tl.exp(gate)
            beta1 = tl.sigmoid(raw_beta)
        elif token_offset == 2:
            q2 = q
            normalized_k2 = normalized_k
            normalized_k_q2 = normalized_k_q
            decay2 = tl.exp(gate)
            beta2 = tl.sigmoid(raw_beta)
        elif token_offset == 3:
            q3 = q
            normalized_k3 = normalized_k
            normalized_k_q3 = normalized_k_q
            decay3 = tl.exp(gate)
            beta3 = tl.sigmoid(raw_beta)
        elif token_offset == 4:
            q4 = q
            normalized_k4 = normalized_k
            normalized_k_q4 = normalized_k_q
            decay4 = tl.exp(gate)
            beta4 = tl.sigmoid(raw_beta)
        elif token_offset == 5:
            q5 = q
            normalized_k5 = normalized_k
            normalized_k_q5 = normalized_k_q
            decay5 = tl.exp(gate)
            beta5 = tl.sigmoid(raw_beta)
        elif token_offset == 6:
            q6 = q
            normalized_k6 = normalized_k
            normalized_k_q6 = normalized_k_q
            decay6 = tl.exp(gate)
            beta6 = tl.sigmoid(raw_beta)
        else:
            q7 = q
            normalized_k7 = normalized_k
            normalized_k_q7 = normalized_k_q
            decay7 = tl.exp(gate)
            beta7 = tl.sigmoid(raw_beta)

    for pid_v in tl.static_range(V // BV):
        current_offs_v = pid_v * BV + offs_v
        state = tl.load(
            state_ptr
            + state_idx * stride_state_block
            + pid_h * stride_state_head
            + current_offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        ).to(tl.float32)
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            if token_offset < query_len:
                token = bos + token_offset
                v = tl.load(
                    v_ptr
                    + token * stride_v_token
                    + pid_h * V
                    + current_offs_v
                ).to(tl.float32)
                if token_offset == 0:
                    q = q0
                    normalized_k = normalized_k0
                    normalized_k_q = normalized_k_q0
                    decay = decay0
                    beta = beta0
                elif token_offset == 1:
                    q = q1
                    normalized_k = normalized_k1
                    normalized_k_q = normalized_k_q1
                    decay = decay1
                    beta = beta1
                elif token_offset == 2:
                    q = q2
                    normalized_k = normalized_k2
                    normalized_k_q = normalized_k_q2
                    decay = decay2
                    beta = beta2
                elif token_offset == 3:
                    q = q3
                    normalized_k = normalized_k3
                    normalized_k_q = normalized_k_q3
                    decay = decay3
                    beta = beta3
                elif token_offset == 4:
                    q = q4
                    normalized_k = normalized_k4
                    normalized_k_q = normalized_k_q4
                    decay = decay4
                    beta = beta4
                elif token_offset == 5:
                    q = q5
                    normalized_k = normalized_k5
                    normalized_k_q = normalized_k_q5
                    decay = decay5
                    beta = beta5
                elif token_offset == 6:
                    q = q6
                    normalized_k = normalized_k6
                    normalized_k_q = normalized_k_q6
                    decay = decay6
                    beta = beta6
                else:
                    q = q7
                    normalized_k = normalized_k7
                    normalized_k_q = normalized_k_q7
                    decay = decay7
                    beta = beta7
                state, correction, state_q = _kda_prepared_step(
                    state, normalized_k, q, v, decay, beta
                )
                out = state_q + correction * normalized_k_q
                tl.store(
                    out_ptr + token * stride_out_token + pid_h * V + current_offs_v,
                    out,
                )
                tl.store(
                    correction_cache_ptr
                    + pid_b * stride_correction_row
                    + pid_h * stride_correction_head
                    + token_offset * stride_correction_pos
                    + current_offs_v * stride_correction_dim,
                    correction,
                )



@triton.jit
def _ascend_kda_recoverssm_verify_tile_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_row,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_row,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    work_id,
    NUM_HEADS: tl.constexpr,
    PAIR_V_TILES: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    WORKLOAD_BUCKET: tl.constexpr,
):
    value_tiles = tl.cdiv(V, BV)
    pid_v = work_id % value_tiles
    pid_b = work_id // (value_tiles * NUM_HEADS)
    pid_h = (work_id // value_tiles) % NUM_HEADS

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    if K == BK and V % BV == 0 and (SPEC_QUERY_LEN != 2 or PAIR_V_TILES):
        mask_k = None
        mask_v = None
        mask_state = None
    else:
        mask_k = offs_k < K
        mask_v = offs_v < V
        mask_state = mask_v[:, None] & mask_k[None, :]

    if state_idx <= null_block_id:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid if mask_v is None else token_valid & mask_v,
            )
        query_len = tl.zeros((), dtype=tl.int64)

    state_idx = tl.maximum(state_idx, 0)
    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(
        state_ptrs,
        mask=mask_state,
        other=0.0 if mask_state is not None else None,
    ).to(tl.float32)
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    if SPEC_QUERY_LEN == 8 or PAIR_V_TILES:
        dt_bias = tl.load(
            dt_bias_ptr + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)

    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        if token_offset < query_len:
            token = bos + token_offset
            q = tl.load(
                q_ptr + token * stride_q_token + pid_h * K + offs_k,
                mask=mask_k,
                other=0.0 if mask_k is not None else None,
            ).to(tl.float32)
            k = tl.load(
                k_ptr + token * stride_k_token + pid_h * K + offs_k,
                mask=mask_k,
                other=0.0 if mask_k is not None else None,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + token * stride_v_token + pid_h * V + offs_v,
                mask=mask_v,
                other=0.0 if mask_v is not None else None,
            ).to(tl.float32)
            raw_g = tl.load(
                raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
                mask=mask_k,
                other=0.0 if mask_k is not None else None,
            ).to(tl.float32)
            raw_beta = tl.load(
                raw_beta_ptr + token * stride_beta_token + pid_h,
            ).to(tl.float32)

            q *= tl.rsqrt(tl.sum(q * q) + 1e-6) * (K**-0.5)
            if SPEC_QUERY_LEN != 8 and not PAIR_V_TILES:
                dt_bias = tl.load(
                    dt_bias_ptr + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
                ).to(tl.float32)
            updated_state, correction = _kda_recurrent_step(
                state,
                k,
                v,
                raw_g,
                raw_beta,
                dt_bias,
                A,
                lower_bound,
                USE_LOWER_BOUND,
            )
            state = updated_state

            out = tl.sum(state * q[None, :], axis=1)
            tl.store(
                out_ptr + token * stride_out_token + pid_h * V + offs_v,
                out,
                mask=mask_v,
            )

            correction_ptr = (
                correction_cache_ptr
                + pid_b * stride_correction_row
                + pid_h * stride_correction_head
                + token_offset * stride_correction_pos
            )
            tl.store(
                correction_ptr + offs_v * stride_correction_dim,
                correction,
                mask=mask_v,
            )
            if pid_v == 0:
                kg_ptr = (
                    kg_cache_ptr
                    + pid_b * stride_kg_row
                    + pid_h * stride_kg_head
                    + token_offset * stride_kg_pos
                )
                tl.store(
                    kg_ptr + offs_k * stride_kg_dim,
                    k,
                    mask=mask_k,
                )
                tl.store(
                    kg_ptr + (K + offs_k) * stride_kg_dim,
                    raw_g,
                    mask=mask_k,
                )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("kda_recoverssm_verify_ascend"),
    key=["K", "V", "SPEC_QUERY_LEN", "WORKLOAD_BUCKET"],
)
@triton.jit
def _ascend_kda_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_row,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_row,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    BATCH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_VECTOR_CORES: tl.constexpr,
    PAIR_V_TILES: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    WORKLOAD_BUCKET: tl.constexpr,
):
    pid = tl.program_id(0)
    value_tiles = tl.cdiv(V, BV)
    total_work = (
        BATCH * NUM_HEADS if PAIR_V_TILES else BATCH * NUM_HEADS * value_tiles
    )
    if PAIR_V_TILES:
        num_programs = min(NUM_VECTOR_CORES, total_work)
        work_per_core = tl.cdiv(total_work, num_programs)
        first_work = pid * work_per_core
        last_work = min(first_work + work_per_core, total_work)
        step = 1
    else:
        first_work = pid
        last_work = total_work
        step = tl.num_programs(0)
    for work_id in tl.parallel(first_work, last_work, step, bind_sub_block=True):
        if PAIR_V_TILES:
            _ascend_kda_recoverssm_verify_pair_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                raw_g_ptr,
                raw_beta_ptr,
                A_log_ptr,
                dt_bias_ptr,
                state_ptr,
                correction_cache_ptr,
                kg_cache_ptr,
                out_ptr,
                query_start_loc_ptr,
                state_indices_ptr,
                lower_bound,
                null_block_id,
                stride_q_token,
                stride_k_token,
                stride_v_token,
                stride_g_token,
                stride_beta_token,
                stride_state_block,
                stride_state_head,
                stride_state_v,
                stride_state_k,
                stride_correction_row,
                stride_correction_head,
                stride_correction_pos,
                stride_correction_dim,
                stride_kg_row,
                stride_kg_head,
                stride_kg_pos,
                stride_kg_dim,
                stride_out_token,
                stride_query_start_loc,
                stride_state_indices,
                work_id,
                NUM_HEADS,
                K,
                V,
                BV,
                SPEC_QUERY_LEN,
                USE_LOWER_BOUND,
            )
        else:
            _ascend_kda_recoverssm_verify_tile_kernel(
                q_ptr,
                k_ptr,
                v_ptr,
                raw_g_ptr,
                raw_beta_ptr,
                A_log_ptr,
                dt_bias_ptr,
                state_ptr,
                correction_cache_ptr,
                kg_cache_ptr,
                out_ptr,
                query_start_loc_ptr,
                state_indices_ptr,
                lower_bound,
                null_block_id,
                stride_q_token,
                stride_k_token,
                stride_v_token,
                stride_g_token,
                stride_beta_token,
                stride_state_block,
                stride_state_head,
                stride_state_v,
                stride_state_k,
                stride_correction_row,
                stride_correction_head,
                stride_correction_pos,
                stride_correction_dim,
                stride_kg_row,
                stride_kg_head,
                stride_kg_pos,
                stride_kg_dim,
                stride_out_token,
                stride_query_start_loc,
                stride_state_indices,
                work_id,
                NUM_HEADS,
                PAIR_V_TILES,
                K,
                V,
                BK,
                BV,
                SPEC_QUERY_LEN,
                USE_LOWER_BOUND,
                WORKLOAD_BUCKET,
            )


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
    checkpoint_state: torch.Tensor,
    records,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint.

    Request ``b`` reads slot ``state_indices[b]`` of ``checkpoint_state`` and
    writes row ``b`` of ``records``. ``records`` exposes ``correction``
    (``[max_num_reqs, H, Q, V]``, float32) and ``key_gate``
    (``[max_num_reqs, H, Q, 2K]``, activation dtype). Returns per-token KDA
    outputs with shape ``[1, total_tokens, H, V]``.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    correction_cache, kg_cache = records.correction, records.key_gate
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    device = q.device
    if any(
        tensor.device != device
        for tensor in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    block_k = triton.next_power_of_2(key_dim)
    workload_bucket = triton.next_power_of_2(batch * num_heads)
    gate_lower_bound = lower_bound or 0.0
    use_lower_bound = lower_bound is not None
    null_block_id = DEFAULT_NULL_BLOCK_ID
    strides = (
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        kg_cache.stride(0),
        kg_cache.stride(1),
        kg_cache.stride(2),
        kg_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
    )
    pair_v_tiles = spec_query_len in (2, 4, 8) and key_dim == 128 and value_dim == 128
    grid = lambda meta: (  # noqa: E731
        min(
            _vector_core_count(device),
            batch
            * num_heads
            * (1 if pair_v_tiles else triton.cdiv(value_dim, meta["BV"])),
        ),
    )
    _ascend_kda_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        out,
        query_start_loc,
        state_indices,
        gate_lower_bound,
        null_block_id,
        *strides,
        BATCH=batch,
        NUM_HEADS=num_heads,
        NUM_VECTOR_CORES=_vector_core_count(device),
        PAIR_V_TILES=pair_v_tiles,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        SPEC_QUERY_LEN=spec_query_len,
        USE_LOWER_BOUND=use_lower_bound,
        WORKLOAD_BUCKET=workload_bucket,
        enable_auto_blockify=False,
    )
    return out
