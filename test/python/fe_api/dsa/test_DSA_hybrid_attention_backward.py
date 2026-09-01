# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the cuDNN HFA backward decomposition."""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_reference import ref_sparse_attention_forward


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_DSA_hybrid_attention_backward_matches_union_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("The initial HFA backward interface requires SM90")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    s_q, s_kv = 16, 256
    heads, qk_dim, value_dim = 32, 576, 512
    sink_tokens, learned_tokens, local_tokens = 8, 24, 32
    scale = 1.0 / math.sqrt(qk_dim)

    q = torch.randn(s_q, heads, qk_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(s_kv, qk_dim, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.randn(heads, dtype=torch.float32, device="cuda")

    # The explicit branch contains sequence-prefix sink keys plus learned keys.
    # Learned keys are chosen well before every local window, keeping the two
    # HFA branches disjoint as required by the decomposition.
    explicit_row = torch.cat(
        (
            torch.arange(sink_tokens, device="cuda"),
            torch.arange(sink_tokens, sink_tokens + learned_tokens, device="cuda"),
        )
    ).to(torch.int32)
    explicit_indices = explicit_row.expand(s_q, -1).contiguous()
    explicit_lengths = torch.full(
        (s_q,), explicit_indices.shape[-1], dtype=torch.int32, device="cuda"
    )

    first_end = s_kv - s_q + 1
    local_indices = torch.stack(
        [
            torch.arange(first_end + row - local_tokens, first_end + row, device="cuda")
            for row in range(s_q)
        ]
    ).to(torch.int32)
    union_indices = torch.cat((explicit_indices, local_indices), dim=-1)
    union_lengths = torch.full(
        (s_q,), union_indices.shape[-1], dtype=torch.int32, device="cuda"
    )

    q_ref = q.detach().clone().requires_grad_(True)
    kv_ref = kv.detach().clone().requires_grad_(True)
    sink_ref = attn_sink.detach().clone().requires_grad_(True)
    out_full, lse = ref_sparse_attention_forward(
        q_ref,
        kv_ref,
        sink_ref,
        union_indices,
        topk_length=union_lengths,
        softmax_scale=scale,
    )
    # Absorbed MLA uses the first 512 KV channels as V while all 576
    # channels participate in QK.
    out_ref = out_full[..., :value_dim]
    dout = torch.randn_like(out_ref)
    out_ref.backward(dout)

    result = DSA.hybrid_attention_backward_wrapper(
        q,
        kv,
        out_ref.detach(),
        dout,
        lse.detach(),
        attn_sink,
        explicit_indices,
        local_tokens=local_tokens,
        explicit_topk_length=explicit_lengths,
        softmax_scale=scale,
    )

    torch.testing.assert_close(result["dq"], q_ref.grad, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(result["dkv"], kv_ref.grad, atol=7e-2, rtol=7e-2)
    torch.testing.assert_close(result["d_sink"], sink_ref.grad, atol=5e-2, rtol=5e-2)
