# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
from typing import Optional
import aiter
import torch
from torch import nn

from aiter.ops.triton.unified_attention import unified_attention
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache

from atom.utils.attn_metadata import ATOMAttentionMetadata

from vllm.attention.backends.abstract import AttentionType, AttentionImpl

class ATOMAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        rotary_emb: Optional[torch.nn.Module] = None,
    ):
        print('[zejun] ATOM init atom attention forward', flush=True)
        print('[zejun] ATOM init atom attention forward, num_heads = ', num_heads, flush=True)
        print('[zejun] ATOM init atom attention forward, head_size = ', head_size, flush=True)
        print('[zejun] ATOM init atom attention forward, scale = ', scale, flush=True)
        print('[zejun] ATOM init atom attention forward, num_kv_heads = ', num_kv_heads, flush=True)
        print('[zejun] ATOM init atom attention forward, kv_cache_dtype = ', kv_cache_dtype, flush=True)
        print('[zejun] ATOM init atom attention forward, sliding_window = ', sliding_window, flush=True)
        print('[zejun] ATOM init atom attention forward, sinks = ', sinks, flush=True)
        print('[zejun] ATOM init atom attention forward, rotary_emb = ', rotary_emb, flush=True)
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.one_scale = torch.tensor(1.0, dtype=torch.float32)
        self.sinks = sinks
        self.sliding_window = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )
        # TODO: remove the ROPE layer
        self.rotary_emb = rotary_emb

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ATOMAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:

        print('[zejun] ATOM call atom attention forward', flush=True)

        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
        )

        # forward_context = get_forward_context()
        # attn_metadata = forward_context.attn_metadata

        # profiler run
        if attn_metadata is None:
            return output.fill_(0)

        # context = forward_context.context
        # position = forward_context.positions

        context = attn_metadata.context
        position = context.positions

        q = query.view(-1, self.num_heads, self.head_size)
        k = key.view(-1, self.num_kv_heads, self.head_size)
        v = value.view(-1, self.num_kv_heads, self.head_size)

        use_triton_unified_attention = (
            self.sliding_window != (-1, -1) or self.head_size != 128
        )
        print('[zejun] ATOM call atom attention forward, use_triton_unified_attention = ', use_triton_unified_attention, flush=True)

        if attn_metadata.slot_mapping.numel():
            k_cache, v_cache = kv_cache.unbind(0)
            k_scale = layer._k_scale
            v_scale = layer._v_scale
        else:
            k_cache = v_cache = torch.tensor([])
            k_scale = v_scale = None

        assert self.rotary_emb is None or (self.rotary_emb is not None and position is not None)
        if k_cache.numel() and v_cache.numel():
            if use_triton_unified_attention:
                k_scale = v_scale = self.one_scale
                k_cache = k_cache.view(
                    k_cache.shape[0], -1, self.num_kv_heads, self.head_size
                )
                v_cache = v_cache.view(
                    v_cache.shape[0], -1, self.num_kv_heads, self.head_size
                )
                if context.is_prefill or self.rotary_emb is None:
                    if self.rotary_emb is not None:
                        q, k = self.rotary_emb(position, q, k)
                    aiter.reshape_and_cache_flash(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        attn_metadata.slot_mapping,
                        (
                            self.kv_cache_dtype
                            if self.kv_cache_dtype.startswith("fp8")
                            else "auto"
                        ),
                        k_scale,
                        v_scale,
                    )
                else:
                    q, k, k_cache, v_cache = fused_qk_rope_reshape_and_cache(
                        q,
                        k,
                        v,
                        k_cache,
                        v_cache,
                        attn_metadata.slot_mapping,
                        position,
                        self.rotary_emb.cos_cache,
                        self.rotary_emb.sin_cache,
                        k_scale,
                        v_scale,
                        self.rotary_emb.is_neox_style,
                        flash_layout=True,
                        apply_scale=self.kv_cache_dtype.startswith("fp8"),
                        offs=None,
                        q_out=q,
                        k_out=k,
                        output_zeros=False,
                    )
            else:
                if self.rotary_emb is not None:
                    assert position is not None
                    q, k = self.rotary_emb(position, q, k)
                if self.kv_cache_dtype == "fp8":
                    aiter.reshape_and_cache_with_pertoken_quant(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        k_scale,
                        v_scale,
                        attn_metadata.slot_mapping,
                        asm_layout=True,
                    )
                else:
                    aiter.reshape_and_cache(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        attn_metadata.slot_mapping,
                        kv_cache_dtype="auto",
                        k_scale=None,
                        v_scale=None,
                        asm_layout=True,
                    )

        if use_triton_unified_attention:
            descale_shape = (attn_metadata.cu_seqlens_q.shape[0] - 1, k.shape[1])
            if k_cache.numel() and v_cache.numel():
                unified_attention(
                    q,
                    k_cache,
                    v_cache,
                    output,
                    cu_seqlens_q=attn_metadata.cu_seqlens_q,
                    seqused_k=attn_metadata.context_lens,
                    max_seqlen_q=attn_metadata.max_seqlen_q,
                    max_seqlen_k=attn_metadata.max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=None,
                    window_size=self.sliding_window,
                    block_table=attn_metadata.block_tables,
                    softcap=0,
                    q_descale=None,
                    k_descale=self.one_scale.expand(descale_shape),
                    v_descale=self.one_scale.expand(descale_shape),
                    sinks=self.sinks,
                )
        elif context.is_prefill:
            # if context.block_tables is not None:  # prefix cache
            #     k, v = k_cache, v_cache
            output = aiter.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=attn_metadata.cu_seqlens_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                max_seqlen_q=attn_metadata.max_seqlen_q,
                max_seqlen_k=attn_metadata.max_seqlen_k,
                min_seqlen_q=attn_metadata.min_seqlen_q,
                dropout_p=attn_metadata.dropout_p,
                softmax_scale=self.scale,
                causal=True,
                out=output,
            )
        else:  # decode
            aiter.pa_fwd_asm(
                q,
                k_cache,
                v_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.block_tables.stride(0),
                K_QScale=k_scale,
                V_QScale=v_scale,
                out_=output,
                high_precision=0,
            )

        output = output.view(-1, self.num_heads * self.head_size)
        return output
