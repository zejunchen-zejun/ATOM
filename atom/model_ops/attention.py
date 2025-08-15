import torch
from torch import nn
import triton
import triton.language as tl

import aiter
from aiter.paged_attn import PagedAttention
from atom.utils.context import get_context
from flash_attn import flash_attn_with_kvcache


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        max_model_len=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.max_model_len = max_model_len
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None
        self.k_scale = self.v_scale = torch.tensor(1.0, dtype=torch.float32)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            aiter.reshape_and_cache(
                k,
                v,
                k_cache,
                v_cache,
                context.slot_mapping,
                kv_cache_dtype="auto",
                k_scale=None,
                v_scale=None,
                asm_layout=False,
            )
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
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
                softmax_scale=self.scale,
                causal=True,
            )
        else:  # decode

            # print(f"{context.block_tables=}")
            # print(f"{context.context_lens=}")
            # kvcache_block_size = self.k_cache.shape[-2]
            # max_num_blocks = (
            #     self.max_model_len + kvcache_block_size - 1
            # ) // kvcache_block_size
            # o = aiter.pa_fwd_asm(
            #     q.contiguous(),
            #     k_cache,
            #     v_cache,
            #     context.block_tables,
            #     context.context_lens,
            #     max_num_blocks,
            #     K_QScale=self.k_scale,
            #     V_QScale=self.v_scale,
            # )
            # self.k_scale = self.k_scale.to(q.device)
            # self.v_scale = self.v_scale.to(q.device)
            o = PagedAttention.forward_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                max_seq_len=self.max_model_len,
                kv_cache_dtype="auto",
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                alibi_slopes=None,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )

            # o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            #                             cache_seqlens=context.context_lens, block_table=context.block_tables,
            #                             softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
