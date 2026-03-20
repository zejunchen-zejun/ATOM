# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Plugin mode extensions for MLAAttention with sparse MLA support.

In vLLM plugin mode, the execution path is:
  vLLM MLAAttention.forward() → custom op → forward_impl() [PATCHED]
  → ATOM's forward_impl_plugin_mode() → forward_impl_sparse_plugin_mode()

The patched forward_impl (see register.py) redirects to ATOM's impl, which
dispatches to sparse or non-sparse based on topk_indices_buffer presence.
forward_impl_sparse_plugin_mode handles everything end-to-end: RoPE, KV cache
write, Q absorption, topk index conversion, sparse kernel, V up-projection.
"""

import torch
import aiter
from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)
from aiter.mla import mla_decode_fwd
from aiter import fused_qk_rope_concat_and_cache_mla

from atom.config import get_current_atom_config
from atom.plugin.prepare import is_vllm
from atom.utils import envs

import triton
import triton.language as tl

import logging

logger = logging.getLogger("atom")

_MLA_MIN_HEADS = 16  # AITER MLA kernels require at least 16 attention heads


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,  # [num_seq, topk]
    cumsum_ptr,  # [num_seq + 1]
    out_tensor_ptr,  # [max_num_seq * topk]
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
):
    num_tokens = in_tensor.size(0)
    block_size = 64
    num_block_per_row = triton.cdiv(topk, block_size)
    grid = (
        num_tokens,
        num_block_per_row,
    )
    fetch_id_to_ragged_kernel[grid](
        in_tensor, cumsum, out_tensor, in_tensor.stride(0), topk, num_tokens, block_size
    )


class MLASparseAttentionImplPluginModeMethods:
    """
    Container class for sparse MLA plugin mode methods.
    This class cannot be instantiated - it only serves as a namespace for methods
    that will be added to MLAAttentionImpl via decorator.
    """

    def __init__(self):
        raise TypeError(
            "MLASparseAttentionImplPluginModeMethods cannot be instantiated. "
            "It is only used as a method container for the decorator."
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        pass

    def _forward_sparse_bf16_kv(
        self,
        q: torch.Tensor, # [sq, heads, d_qk]
        kv_cache: torch.Tensor, # [blocks, heads, d_qk]
        topk_indices_global: torch.Tensor, # [sq, topk]
        attn_metadata,
        layer,
    ) -> torch.Tensor:
        sparse_meta = attn_metadata.plugin_metadata

        num_tokens = q.shape[0]
        padded_num_heads = max(self.num_heads, _MLA_MIN_HEADS)
        output = torch.empty(
            [num_tokens, padded_num_heads, self.kv_lora_rank],
            dtype=q.dtype,
            device=q.device,
        )

        # Build ragged layout only once per forward as it shared across MLA layers
        if not getattr(sparse_meta, "ragged_layout_built", False):
            seq_len = (topk_indices_global != -1).sum(dim=-1)
            torch.cumsum(seq_len, dim=0, out=sparse_meta.paged_kv_indptr[1:])
            sparse_meta.paged_kv_indptr_rest.fill_(sparse_meta.paged_kv_indptr[-1])
            fetch_id_to_ragged_triton(
                topk_indices_global,
                sparse_meta.paged_kv_indptr,
                sparse_meta.paged_kv_indices,
                sparse_meta.topk_tokens,
            )
            sparse_meta.ragged_layout_built = True

        kv_buffer = kv_cache.unsqueeze(2)
        
        fp8_kv = self.kv_cache_dtype.startswith("fp8")
        if fp8_kv and sparse_meta.work_meta_data is not None:
            # Update persistent scheduling metadata for this batch
            from aiter.ops.attention import decode_update_mla_metadata_v1
            decode_update_mla_metadata_v1(
                seqlens_qo_indptr=sparse_meta.qo_indptr,
                seqlens_kv_indptr=sparse_meta.paged_kv_indptr,
                kv_last_page_lens=sparse_meta.paged_kv_last_page_len,
                num_heads_per_head_k=padded_num_heads,
                num_heads_k=1,
                is_causal=False,
                work_metadata_ptrs=sparse_meta.work_meta_data,
                work_info_set=sparse_meta.work_info_set,
                work_indptr=sparse_meta.work_indptr,
                reduce_indptr=sparse_meta.reduce_indptr,
                reduce_final_map=sparse_meta.reduce_final_map,
                reduce_partial_map=sparse_meta.reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=1,
            )

            mla_decode_fwd(
                q,
                kv_buffer.view(-1, 1, 1, q.shape[-1]),
                output,
                sparse_meta.qo_indptr,
                sparse_meta.paged_kv_indptr,
                sparse_meta.paged_kv_indices,
                sparse_meta.paged_kv_last_page_len,
                1,  # max_qo_len
                sm_scale=self.scale,
                q_scale=layer._q_scale,
                kv_scale=layer._k_scale,
                page_size=1,
                # Persistent mode args:
                work_meta_data=sparse_meta.work_meta_data,
                work_indptr=sparse_meta.work_indptr,
                work_info_set=sparse_meta.work_info_set,
                reduce_indptr=sparse_meta.reduce_indptr,
                reduce_final_map=sparse_meta.reduce_final_map,
                reduce_partial_map=sparse_meta.reduce_partial_map,
            )
            
        else:
            use_persistent_mode = not (
                self.dcp_world_size > 1 and self.kv_cache_dtype == "fp8"
            )
            if not use_persistent_mode:
                # DP : disable persistent mode to avoid overflow
                work_meta_data = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
            else:
                work_meta_data = attn_metadata.work_meta_data
                work_indptr = attn_metadata.work_indptr
                work_info_set = attn_metadata.work_info_set
                reduce_indptr = attn_metadata.reduce_indptr
                reduce_final_map = attn_metadata.reduce_final_map
                reduce_partial_map = attn_metadata.reduce_partial_map

            mla_decode_fwd(
                q,
                kv_buffer.view(-1, 1, 1, q.shape[-1]),
                output,
                sparse_meta.qo_indptr,
                sparse_meta.paged_kv_indptr,
                sparse_meta.paged_kv_indices,
                sparse_meta.paged_kv_last_page_len,
                1,  # max_qo_len = 1 for sparse MQA (each token is a single query)
                work_meta_data=work_meta_data,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                sm_scale=self.scale,
                q_scale=layer._q_scale,
                kv_scale=layer._k_scale,
                # page_size=1,
            )

        if self.num_heads < _MLA_MIN_HEADS:
            head_repeat_factor = _MLA_MIN_HEADS // self.num_heads
            output = output[:, :: head_repeat_factor, :].contiguous()

        return output[:, :self.num_heads, :]

    def forward_impl_sparse_plugin_mode(
        self,
        layer,
        q,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output,
    ):
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        from vllm.distributed.parallel_state import get_dcp_group
        from vllm.platforms import current_platform

        if self.dcp_world_size == -1:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")
        
        sparse_meta = attn_metadata.plugin_metadata

        num_actual_toks = sparse_meta.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...].unsqueeze(1)
        
        if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        atom_config = get_current_atom_config()
        positions = atom_config.compilation_config.static_forward_context["positions"][
            :num_actual_toks
        ]

        fp8_attention = self.kv_cache_dtype.startswith("fp8")
        if fp8_attention:
            from vllm.platforms import current_platform
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        # Q absorption: q_nope -> W_K BMM -> ql_nope, then concat with q_pe
        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)

        if self.q_pad_num_heads is not None:
            B, N, L = q_pe.shape
            pe_padded = q_pe.new_empty((B, self.q_pad_num_heads, L))
            pe_padded.resize_((B, N, L))
            pe_padded.copy_(q_pe)
            q_pe = pe_padded

        if self.is_aiter_triton_fp4_bmm_enabled:
            ql_nope = batched_gemm_a16wfp4(
                q_nope,
                self.W_K,
                self.W_K_scale,
                transpose_bm=True,
                prequant=True,
                y_scale=layer._q_scale if fp8_attention else None,
            )
        else:
            # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
            ql_nope = _aiter_triton_fp8_bmm(
                q_nope,
                self.W_K,
                self.W_K_scale,
                group_size=128,
                transpose_bm=True,
            )

        q_out = torch.empty(
            (
                ql_nope.shape[0],
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=ql_nope.dtype,
            device=ql_nope.device,
        )
        if kv_cache.numel() > 0:
            fused_qk_rope_concat_and_cache_mla(
                ql_nope,
                q_pe,
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache.view(
                    kv_cache.shape[0], -1, self.kv_lora_rank + self.qk_rope_head_dim
                ),
                q_out,
                attn_metadata.slot_mapping,
                self._k_scale,
                self._q_scale,
                positions,
                self.rotary_emb.cos_cache,
                self.rotary_emb.sin_cache,
                is_neox=self.rotary_emb.is_neox_style,
                is_nope_first=True,
            )
            
        head_repeat_factor = _MLA_MIN_HEADS // self.num_heads
        if head_repeat_factor > 1:
            q_out = q_out.repeat_interleave(head_repeat_factor, dim=1)

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        try:
            from vllm.v1.attention.backends.mla.sparse_utils import (
                triton_convert_req_index_to_global_index,
            )
        except ImportError:
            from vllm.v1.attention.backends.mla.flashmla_sparse import (
                triton_convert_req_index_to_global_index,
            )

        req_id_i32 = sparse_meta.req_id_per_token.to(dtype=torch.int32)
        block_table_i32 = sparse_meta.block_table.to(dtype=torch.int32)
        topk_indices_i32 = topk_indices.to(dtype=torch.int32)
        topk_indices_global = triton_convert_req_index_to_global_index(
            req_id_i32, # sparse_meta.req_id_per_token,
            block_table_i32, # sparse_meta.block_table,
            topk_indices_i32, # topk_indices,
            BLOCK_SIZE=sparse_meta.block_size,
            NUM_TOPK_TOKENS=sparse_meta.topk_tokens,
        )
        attn_out = self._forward_sparse_bf16_kv(
            q_out, kv_cache, topk_indices_global, attn_metadata, layer
        )

        # V up-projection
        out_up_proj = self._v_up_proj(attn_out)
        output[:num_actual_toks] = out_up_proj

        return output_padded


# ---------------------------------------------------------------------------
# Sparse MLA plugin mode initialization
# ---------------------------------------------------------------------------


def _mla_sparse_plugin_mode_init(self, *args, **kwargs):
    """Extra initialization for MLAAttentionImpl in sparse plugin mode (vllm)."""
    if is_vllm():
        self.supports_quant_query_input = False
        self.dcp_world_size: int = -1
        self.is_aiter_triton_fp4_bmm_enabled = (
            envs.ATOM_USE_TRITON_MXFP4_BMM
            and self.kv_b_proj.weight.dtype == torch.bfloat16
        )
        self.q_pad_num_heads = kwargs.get("q_pad_num_heads", None)

        # Sparse MLA specific: verify topk_indices_buffer is present
        assert getattr(self, "topk_indices_buffer", None) is not None, (
            "topk_indices_buffer must be set for sparse MLA plugin mode. "
            "Ensure the model's Indexer is properly initialized."
        )


# ---------------------------------------------------------------------------
# Decorator for injecting sparse MLA methods
# ---------------------------------------------------------------------------


def MLASparseAttentionImplDecoratorForPluginMode(cls):
    """
    Decorator that injects sparse MLA methods into the MLAAttentionImpl class.
    Applied alongside the regular MLAAttentionImplDecoratorForPluginMode.

    Injects forward_impl_sparse_plugin_mode and _forward_sparse_bf16_kv.
    The patched forward_impl in register.py calls forward_impl_plugin_mode,
    which dispatches to forward_impl_sparse_plugin_mode when topk_indices_buffer
    is set.
    """
    sparse_method_names = [
        "_forward_sparse_bf16_kv",
        "forward_impl_sparse_plugin_mode",
    ]

    logger.info(
        "Use MLASparseAttentionImplDecoratorForPluginMode to decorate MLAAttention"
    )

    # Add all sparse methods to the target class
    for method_name in sparse_method_names:
        method = getattr(MLASparseAttentionImplPluginModeMethods, method_name)
        setattr(cls, method_name, method)

    # Wrap __init__ to inject sparse plugin-mode initialization
    orig_init = cls.__init__

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Only run sparse init if this instance has a topk_indices_buffer
        # (i.e., the model uses sparse MLA)
        if getattr(self, "topk_indices_buffer", None) is not None:
            _mla_sparse_plugin_mode_init(self, *args, **kwargs)

    cls.__init__ = new_init

    return cls
