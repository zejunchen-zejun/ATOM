from dataclasses import dataclass
from typing import Type, Optional

from atom.utils.forward_context import AttentionMetaData, Context
from .backends import CommonAttentionBuilder, AttentionBackend
import torch
import numpy as np

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import Attention


class AiterBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["AiterAttentionMetadataBuilder"]:
        return AiterAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["Attention"]:
        return Attention

class AiterAttentionMetadataBuilder(CommonAttentionBuilder):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

    def __init__(self, block_size: int):
        super().__init__(block_size)  # Call parent __init__ to initialize _cached_kv_cache_data

    def prepare_decode(self, batch: ScheduledBatch, bs: int, forward_vars):
        scheduled_bs = batch.total_seqs_num_decode
        seqs = list(batch.seqs.values())
        self.total_blocks = 0
        dropout_p = 0.0
        max_q_len = 1

        context_lens = [seq.num_tokens for seq in seqs]
        positions = context_lens
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            for seq in seqs
        ]
        slot_mapping.extend([-1] * (bs - scheduled_bs))

        block_tables = forward_vars["block_tables"].np
        for i, seq in enumerate(seqs):
            block_tables[i] = 0
            block_tables[i, : seq.num_blocks] = seq.block_table

        sum_scheduled_tokens = batch.total_tokens_num_decode
        var = forward_vars
        var["slot_mapping"].np[:bs] = slot_mapping
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("block_tables", bs),
        ]

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            **ctx,
        )

        positions=var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, forward_vars, bs: int) -> AttentionMetaData:
        attn_matadata = AttentionMetaData(
            slot_mapping=forward_vars["slot_mapping"].gpu[:bs],
            context_lens=forward_vars["context_lens"].gpu[:bs],
            block_tables=forward_vars["block_tables"].gpu[:bs],
            max_q_len=1,
            cu_seqlens_q=forward_vars["cu_seqlens_q"].gpu[: bs + 1],
        )
        positions=forward_vars["positions"].copy_to_gpu(bs)
        context = Context(positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs)
        return attn_matadata, context
