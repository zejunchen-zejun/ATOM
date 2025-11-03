from dataclasses import dataclass
from typing import Type, Optional

from atom.config import KVCacheConfig, KVCacheTensor
from atom.utils.forward_context import AttentionMetaData, Context
from .backends import CommonAttentionBuilder, AttentionBackend
import torch
import numpy as np

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention


class AiterMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention


class AiterMLAMetadataBuilder(CommonAttentionBuilder):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

    def __init__(self, block_size: int):
        super().__init__(block_size)  # Call parent __init__ to initialize _cached_kv_cache_data
        assert self.block_size == 1, "AITER MLA requires only block size 1."

    def prepare(self, batch: ScheduledBatch):
        self.paged_kv_indices: list[int] = []
        self.paged_kv_indptr: list[int] = [0]
        self.paged_kv_last_page_lens: list[int] = []
        self.total_blocks = 0

        seqs = list(batch.seqs.values())
        for seq in seqs:
            current_seq_len = seq.num_tokens
            self._update_paged_kv_tensors(seq.block_table, current_seq_len)

    def _update_paged_kv_tensors(self, block_table: list[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        self.total_blocks += len(block_table)
        block_table_bound = (
            seq_len // self.block_size + 1
            if seq_len % self.block_size != 0
            else seq_len // self.block_size
        )
        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] + block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_lens.append(last_page_len)
    

    def prepare_decode(self, batch: ScheduledBatch, bs: int, forward_vars):
        scheduled_bs = batch.total_seqs_num_decode
        seqs = list(batch.seqs.values())
        dropout_p = 0.0
        max_q_len = 1
        self.total_blocks = 0

        context_lens = [seq.num_tokens for seq in seqs]
        positions = context_lens
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            for seq in seqs
        ]
        slot_mapping.extend([-1] * (bs - scheduled_bs))

        self.prepare(batch)


        sum_scheduled_tokens = batch.total_tokens_num_decode
        var = forward_vars
        var["slot_mapping"].np[:bs] = slot_mapping
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
        ]

        if len(self.paged_kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the scheduler
            self.paged_kv_indices.extend([0] * (self.total_blocks - len(self.paged_kv_indices)))
            var["kv_indices"].np[: self.total_blocks] = np.array(
                self.paged_kv_indices, dtype=np.int64
            )
            var["kv_indptr"].np[: scheduled_bs + 1] = np.array(
                self.paged_kv_indptr, dtype=np.int64
            )
            var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = var["kv_indptr"].np[
                scheduled_bs
            ]
            var["kv_last_page_lens"].np[:scheduled_bs] = np.array(
                self.paged_kv_last_page_lens, dtype=np.int64
            )
            var["kv_last_page_lens"].np[scheduled_bs:bs] = 0
            vars_used = [
                ("slot_mapping", bs),  # TODO: MTP support
                ("context_lens", bs),
                # ("block_tables", bs),
                ("cu_seqlens_q", bs + 1),
                ("kv_indices", sum(context_lens)),
                ("kv_indptr", bs + 1),
                ("kv_last_page_lens", bs),
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
            kv_indptr=forward_vars["kv_indptr"].gpu[: bs + 1],
            kv_indices=forward_vars["kv_indices"].gpu[:],
            kv_last_page_lens=forward_vars["kv_last_page_lens"].gpu[:bs],
        )
        positions=forward_vars["positions"].copy_to_gpu(bs)
        context = Context(positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs)
        return attn_matadata, context
