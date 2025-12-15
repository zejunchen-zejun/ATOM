# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type, ClassVar

import torch
from atom.model_ops.attention_mha import ATOMAttentionImpl
from atom.utils.attn_metadata import Context, ATOMAttentionMetadata
from atom.utils import CpuGpuBuffer

# from .backends import AttentionBackend
# TODO: for MLA, the attn backend should use vllm
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.abstract import MultipleOf

from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.config.vllm import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

# TODO: use vllm father class
class ATOMAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> Type["ATOMAttentionImpl"]:
        return ATOMAttentionImpl

    @staticmethod
    def get_builder_cls() -> Type["ATOMAttentionMetadataBuilder"]:
        return ATOMAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")

        return (2, num_blocks, block_size, num_kv_heads, head_size)

class ATOMAttentionMetadataBuilder(AttentionMetadataBuilder[ATOMAttentionMetadata]):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]
    # TODO: add cudagraph support
    _cudagraph_support = AttentionCGSupport.NEVER
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None
        self.total_tokens: int = 0

        # for recording position info
        i64_kwargs = {"dtype": torch.int64, "device": device}
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.positions = CpuGpuBuffer(max_num_batched_tokens, **i64_kwargs),

    def build_for_cudagraph_capture(self, bs: int) -> ATOMAttentionMetadata:
        assert False, "Not implemented for cuda graph capture for now"
        var = self.model_runner.forward_vars
        attn_matadata = ATOMAttentionMetadata(
            slot_mapping=var["slot_mapping"].gpu[:bs],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_q_len=1,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            max_seqlen_k=self.model_runner.config.max_model_len,
        )
        positions = var["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        attn_matadata.context = context
        return attn_matadata

    def build_decode(self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False
    ):
        bs = common_attn_metadata.num_reqs
        # self.total_blocks = 0
        dropout_p = 0.0
        max_q_len = 1
        min_seqlen_q = 0

        context_lens = common_attn_metadata.seq_lens_cpu
        max_seqlen_k = max(context_lens)
        positions = [i - 1 for i in context_lens]
        # slot_mapping = [
        #     block_table[-1] * self.block_size + last_block_num - 1
        #     for block_table, last_block_num in zip(
        #         batch.block_tables, batch.last_block_num_tokens
        #     )
        # ]
        # slot_mapping.extend([-1] * (bs - scheduled_bs))

        # self.prepare_block_tables(batch)

        # var = self.model_runner.forward_vars

        # for decode, sum scheduled tokens is equal to bs
        sum_scheduled_tokens = bs
        cu_seqlens_q = common_attn_metadata.query_start_loc - common_attn_metadata.query_start_loc[0]
        # sum_scheduled_tokens = batch.total_tokens_num_decode

        # var["slot_mapping"].np[:bs] = slot_mapping
        # var["positions"].np[:sum_scheduled_tokens] = positions
        self.positions.np[:sum_scheduled_tokens] = positions

        # var["context_lens"].np[:bs] = context_lens
        # vars_used = [
        #     # ("slot_mapping", bs),  # TODO: MTP support
        #     ("context_lens", bs),
        #     # ("block_tables", bs),
        # ]

        # if self.has_sliding_window:
            # vars_used.append(("cu_seqlens_q", bs + 1))
        # ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}

        attn_metadata = ATOMAttentionMetadata(
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_tables=common_attn_metadata.block_table_tensor,
            context=Context(
                positions=self.positions.copy_to_gpu(sum_scheduled_tokens),
                is_prefill=False,
                batch_size=bs,
                graph_bs=bs
            ),
            cu_seqlens_q=cu_seqlens_q,
            context_lens=common_attn_metadata.seq_lens,
        )

        # positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return attn_metadata

    def build_prefill(self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False
    ):
        bs = common_attn_metadata.num_reqs
        sum_scheduled_tokens = common_attn_metadata.num_actual_tokens

        # var = forward_vars
        positions = []
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        context_lens = common_attn_metadata.seq_lens_cpu
        num_cached_tokens = common_attn_metadata.num_computed_tokens_cpu

        for i in range(bs):
            seqlen = context_lens[i]
            cached_seqlen = num_cached_tokens[i]
            positions.extend(list(range(cached_seqlen, seqlen)))
            seqlen_q = seqlen - cached_seqlen
            seqlen_k = seqlen
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # num_blocks = (seqlen + self.block_size - 1) // self.block_size
            # num_cached_blocks = (cached_seqlen + self.block_size - 1) // self.block_size
            # last_block_tokens = batch.last_block_num_tokens[i]
            # block_table = batch.block_tables[i]
            # for i in range(num_cached_blocks, num_blocks):
            #     start = block_table[i] * self.block_size
            #     if i != num_blocks - 1:
            #         end = start + self.block_size
            #     else:
            #         end = start + last_block_tokens
            #     slot_mapping.extend(list(range(start, end)))

        # if cu_seqlens_k[-1] > batch.total_tokens_num:  # prefix cache
        #     self.prepare_block_tables(batch)
        # var["positions"].np[:sum_scheduled_tokens] = positions
        self.positions.np[:sum_scheduled_tokens] = positions
        # var["slot_mapping"].np[: len(slot_mapping)] = slot_mapping
        
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)

        # for now, all requests are prefill
        cu_seqlens_q = common_attn_metadata.query_start_loc - common_attn_metadata.query_start_loc[0]

        min_seqlen_q = 0
        dropout_p = 0.0
        # vars_used = [
        #     ("cu_seqlens_q", bs + 1),
        #     ("slot_mapping", len(slot_mapping)),
        # ]

        # TODO: support sliding window
        # if self.has_sliding_window:
        #     var["context_lens"].np[:bs] = batch.context_lens[:bs]
        #     vars_used.append(("context_lens", bs))
        #     self.prepare_block_tables(batch)
        #     vars_used.append(("block_tables", bs))

        # ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        attn_metadata = ATOMAttentionMetadata(
            cu_seqlens_k=cu_seqlens_k.cuda(non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            block_tables=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            context=Context(
                positions=self.positions.copy_to_gpu(sum_scheduled_tokens),
                is_prefill=True,
                batch_size=bs,
                graph_bs=bs
            ),
            cu_seqlens_q=cu_seqlens_q,
            context_lens=common_attn_metadata.seq_lens,
        )
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "ATOMAttentionMetadata":

        # TODO: for now chunked prefill doesn't support
        # TODO: disable the chunked prefill
        print('[zejun] ATOM ATOMAttentionMetadataBuilder build', flush=True)
        _build_prefill = common_attn_metadata.max_query_len > 1
        if _build_prefill:
            print('[zejun] ATOM ATOMAttentionMetadataBuilder build prefill', flush=True)
            return self.build_prefill(common_prefix_len, common_attn_metadata, fast_build)
        else:
            print('[zejun] ATOM ATOMAttentionMetadataBuilder build decode', flush=True)
            return self.build_decode(common_prefix_len, common_attn_metadata, fast_build)
