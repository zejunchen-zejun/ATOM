# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type, ClassVar

import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import Attention
from atom.utils.forward_context import ATOMAttentionMetadata, Context
from vllm.vllm.vllm.inputs import data

from .backends import AttentionBackend, CommonAttentionBuilder
# TODO: for MLA, the attn backend should use vllm
# from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.abstract import MultipleOf
from vllm.attention.backends.registry import (AttentionBackendEnum,
                                              register_backend)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    CommonAttentionMetadata
)

# TODO: use vllm father class
@register_backend(AttentionBackendEnum.CUSTOM, "atom.model_ops.attentions.aiter_attention.ATOMAttentionBackend")
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
        return "ATOM_ATTENTION"

    @staticmethod
    def get_impl_cls() -> Type["Attention"]:
        return Attention

    @staticmethod
    def get_builder_cls() -> Type["ATOMAttentionMetadataBuilder"]:
        return ATOMAttentionMetadataBuilder


@dataclass
class ATOMAttentionMetadata:


class ATOMAttentionMetadataBuilder(CommonAttentionBuilder):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]
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

        # self.extend_workspace = torch.empty(
        #     [2, _CP_TOKENS_PER_ITER_ROCM, self.num_heads_kv, self.headdim],
        #     dtype=self.model_config.dtype,
        #     device=device,
        # )

    def build_for_cudagraph_capture(self, bs: int) -> ATOMAttentionMetadata:
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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "ATOMAttentionMetadata":



    # def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        dropout_p = 0.0
        max_q_len = 1
        min_seqlen_q = 0

        context_lens = batch.context_lens
        max_seqlen_k = max(context_lens)
        positions = [i - 1 for i in context_lens]
        slot_mapping = [
            block_table[-1] * self.block_size + last_block_num - 1
            for block_table, last_block_num in zip(
                batch.block_tables, batch.last_block_num_tokens
            )
        ]
        slot_mapping.extend([-1] * (bs - scheduled_bs))

        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[:bs] = slot_mapping
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("block_tables", bs),
        ]
        if self.has_sliding_window:
            vars_used.append(("cu_seqlens_q", bs + 1))
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        attn_metadata = ATOMAttentionMetadata(
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            **ctx,
        )

        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return attn_metadata, positions

