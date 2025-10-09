# from flash_attn import flash_attn_with_kvcache
from dataclasses import dataclass

import aiter
import torch
import triton
import triton.language as tl
from aiter.paged_attn import PagedAttention
from torch import nn

from atom.utils.context import get_context
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import (
    AttentionMetadata,
    ForwardContext,
    get_forward_context,
    set_forward_context,
)


# Dynamo will not try to inspect any of the internal operations for prefill or decode
# @torch.library.custom_op("aiter::unified_attention_with_output", mutates_args=["q", "k", "v", "k_cache", "v_cache", "k_scale", "v_scale"])
def unified_attention_with_output(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float, kv_cache_dtype: str, layer_num: int
)-> torch.Tensor:
    context = get_context()
    if context.slot_mapping.numel():
        # not dummy run
        forward_context: ForwardContext = get_forward_context()
        attn_metadata_ = forward_context.no_compile_layers[layer_num]
        k_cache = attn_metadata_.k_cache
        v_cache = attn_metadata_.v_cache
        k_scale = attn_metadata_.k_scale
        v_scale = attn_metadata_.v_scale
    else:
        # dummy run before allocate kv_cache, thus we create manually
        k_cache = v_cache = torch.tensor([])
        k_scale = v_scale = None

    if k_cache.numel() and v_cache.numel():
        if kv_cache_dtype == "fp8":
            aiter.reshape_and_cache_with_pertoken_quant(
                k,
                v,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                context.slot_mapping,
                asm_layout=True,
            )
        else:
            aiter.reshape_and_cache(
                k,
                v,
                k_cache,
                v_cache,
                context.slot_mapping,
                kv_cache_dtype="auto",
                k_scale=None,
                v_scale=None,
                asm_layout=True,
            )


    if context.is_prefill:
        # if context.block_tables is not None:  # prefix cache
        #     k, v = k_cache, v_cache
        o = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            min_seqlen_q=context.min_seqlen_q,
            dropout_p=context.dropout_p,
            softmax_scale=scale,
            causal=True,
        )
    else:  # decode
        o = aiter.pa_fwd_asm(
            q,
            k_cache,
            v_cache,
            context.block_tables,
            context.context_lens,
            context.block_tables.stride(0),
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=None,
            high_precision=0,
        )

    return o

# @unified_attention_with_output.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float, kv_cache_dtype: str, layer_num: int
)-> torch.Tensor:
    output_shape = q.shape
    output = torch.zeros(output_shape,
                            dtype=q.dtype,
                            device=q.device)
    return output


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=[],
    fake_impl=_,
)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.layer_num = layer_num

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        

        o = torch.ops.aiter.unified_attention_with_output(q, k, v, 
                    self.scale, self.kv_cache_dtype, self.layer_num)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
