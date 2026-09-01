# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid fixed-window/sparse absorbed-MLA backward.

The operator implements the HFA backward decomposition used by DSA streaming
indexing.  The explicit sink/learned branch reuses the native DSA sparse
backward, while the causal-local branch reuses its SM90 576/512 pipeline with
an implicit bottom-right Q2 window mapping. Both branches consume the *full hybrid*
output and LSE, so their gradient contributions use one shared softmax
normalizer and can be added directly. The sparse branch's backward preprocess
is reused by the local branch.

This first interface intentionally covers the unpacked, batch-one bottom-right
chunk used by absorbed MLA training.  Packed/ragged dispatch must use a future
THD specialization rather than silently applying the wrong window alignment.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict
from cudnn.deepseek_sparse_attention.utils.runtime import torch_stream_context

from ..sparse_attention_backward import sparse_attention_backward_wrapper


def _run_local_window_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    metadata_indices: torch.Tensor,
    *,
    local_tokens: int,
    softmax_scale: float,
    precomputed_dpsum: Optional[torch.Tensor] = None,
    precomputed_lse_log2: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ..sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90

    # Reuse the native DSA 576/512 backward pipeline. In implicit mode its M64
    # tile packs two 32-head adjacent queries, loads their local KV union once,
    # masks the two shifted causal windows, and scatters one combined dKV tile.
    return flash_attn_bwd_sm90(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        topk_idxs=metadata_indices,
        topk_length=None,
        need_d_sink=False,
        implicit_local_tokens=local_tokens,
        precomputed_dpsum=precomputed_dpsum,
        precomputed_lse_log2=precomputed_lse_log2,
    )


class HybridAttentionBackward(APIBase):
    """Experimental HFA backward using cuDNN explicit and implicit DSA branches."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_kv: torch.Tensor,
        sample_out: torch.Tensor,
        sample_dout: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_attn_sink: torch.Tensor,
        sample_explicit_topk_idxs: torch.Tensor,
        *,
        local_tokens: int,
        sample_explicit_topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.dout_desc = self._make_tensor_desc(sample_dout, name="sample_dout")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.indices_desc = self._make_tensor_desc(sample_explicit_topk_idxs, name="sample_explicit_topk_idxs")
        self.length_desc = self._make_tensor_desc(sample_explicit_topk_length, name="sample_explicit_topk_length")
        self.local_tokens = int(local_tokens)
        self.softmax_scale = softmax_scale

    def check_support(self) -> bool:
        major, _ = torch.cuda.get_device_capability()
        self._runtime_error_if(major < 9, f"HybridAttentionBackward requires SM90+, found SM{major}")
        self._runtime_error_if(major != 9, "The initial HFA backward interface currently supports SM90 only")
        self._value_error_if(self.q_desc.ndim != 3, f"Q must be [S_q,H,D], got {self.q_desc.shape}")
        self._value_error_if(self.kv_desc.ndim != 2, f"KV must be [S_kv,D], got {self.kv_desc.shape}")
        self._value_error_if(self.indices_desc.ndim != 2, "explicit_topk_idxs must be [S_q,K_explicit]")
        self._value_error_if(self.local_tokens <= 0, "local_tokens must be positive")
        self._value_error_if(
            self.q_desc.shape[1] != 32,
            "SM90 implicit-local Q2 backward currently requires 32 query heads",
        )
        self._value_error_if(
            self.q_desc.shape[0] % 2 != 0,
            "SM90 implicit-local Q2 backward requires an even query length",
        )
        self._value_error_if(self.q_desc.shape[-1] != self.kv_desc.shape[-1], "Q and KV QK widths must match")
        self._value_error_if(self.out_desc.shape != self.dout_desc.shape, "out and dout shapes must match")
        self._value_error_if(self.out_desc.shape[:2] != self.q_desc.shape[:2], "out must match Q rows and heads")
        self._value_error_if(self.lse_desc.shape != self.q_desc.shape[:2], "lse must be [S_q,H]")
        self._value_error_if(self.attn_sink_desc.shape != (self.q_desc.shape[1],), "attn_sink must be [H]")
        self._value_error_if(self.indices_desc.shape[0] != self.q_desc.shape[0], "explicit indices need one row per query")
        self._value_error_if(
            self.kv_desc.shape[0] - self.q_desc.shape[0] + 1 < self.local_tokens,
            "the first bottom-right query has fewer valid keys than local_tokens",
        )
        self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        self._check_dtype(self.kv_desc, self.q_desc.dtype, name="KV")
        self._check_dtype(self.out_desc, self.q_desc.dtype, name="out")
        self._check_dtype(self.dout_desc, self.q_desc.dtype, name="dout")
        self._check_dtype(self.lse_desc, torch.float32, name="lse")
        self._check_dtype(self.attn_sink_desc, torch.float32, name="attn_sink")
        self._check_dtype(self.indices_desc, torch.int32, name="explicit_topk_idxs")
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        # Both reused backends own shape-specialized compile caches and are
        # lazily primed with real storage during execute().
        self._compiled_kernel = True

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
        explicit_topk_idxs: torch.Tensor,
        *,
        explicit_topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        with torch_stream_context(current_stream):

            def sparse_branch(*, return_preprocess_aux: bool = False):
                result = sparse_attention_backward_wrapper(
                    q,
                    kv,
                    out,
                    dout,
                    lse,
                    attn_sink,
                    explicit_topk_idxs,
                    softmax_scale=scale,
                    topk_length=explicit_topk_length,
                    return_preprocess_aux=return_preprocess_aux,
                )
                branch_outputs = (
                    result["dq"],
                    result["dkv"],
                    result["d_sink"],
                )
                if return_preprocess_aux:
                    branch_outputs = (
                        *branch_outputs,
                        result["dpsum"],
                        result["lse_log2"],
                    )
                return branch_outputs

            # The two branches share full-attention O/dO/LSE. Export the sparse
            # branch's preprocess results and reuse them in the local branch
            # instead of repeating the O*dO reduction and LSE conversion.
            sparse_with_aux = sparse_branch(return_preprocess_aux=True)
            sparse_result = sparse_with_aux[:3]
            dpsum, lse_log2 = sparse_with_aux[3:]
            local_result = _run_local_window_backward(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                explicit_topk_idxs,
                local_tokens=self.local_tokens,
                softmax_scale=scale,
                precomputed_dpsum=dpsum,
                precomputed_lse_log2=lse_log2,
            )

            dq_sparse, dkv_sparse, d_sink = sparse_result
            dq_local, dkv_local = local_result
            dq_local.add_(dq_sparse)
            dkv_local.add_(dkv_sparse)
            return dq_local, dkv_local, d_sink


_cache: dict[tuple, HybridAttentionBackward] = {}


def hybrid_attention_backward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    explicit_topk_idxs: torch.Tensor,
    *,
    local_tokens: int,
    explicit_topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Run paper-style HFA backward for an unpacked bottom-right query chunk.

    ``out`` and ``lse`` must come from the complete union of explicit sparse
    and implicit local keys.  ``explicit_topk_idxs`` must contain sink and
    learned keys only; local keys are generated by the cuDNN window mask.
    """
    key = (
        q.dtype,
        tuple(q.shape),
        tuple(kv.shape),
        tuple(out.shape),
        tuple(explicit_topk_idxs.shape),
        explicit_topk_length is not None,
        int(local_tokens),
        softmax_scale,
    )
    obj = _cache.get(key)
    if obj is None:
        obj = HybridAttentionBackward(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            explicit_topk_idxs,
            local_tokens=local_tokens,
            sample_explicit_topk_length=explicit_topk_length,
            softmax_scale=softmax_scale,
        )
        assert obj.check_support()
        obj.compile()
        _cache[key] = obj

    dq, dkv, d_sink = obj.execute(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        explicit_topk_idxs,
        explicit_topk_length=explicit_topk_length,
        softmax_scale=softmax_scale,
        current_stream=stream,
    )
    return TupleDict(dq=dq, dkv=dkv, d_sink=d_sink)
