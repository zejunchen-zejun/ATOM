# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Optional, Set, Dict

import torch

@dataclass
class Context:
    # This context is used to store the basic context of the forward.
    positions: torch.Tensor
    is_prefill: bool = False
    batch_size: int = 0
    graph_bs: int = 0

    def __init__(
        self,
        positions: torch.Tensor,
        is_prefill: bool = False,
        batch_size: int = 0,
        graph_bs: int = 0,
    ):
        self.positions = positions
        self.is_prefill = is_prefill
        self.batch_size = batch_size
        self.graph_bs = graph_bs


@dataclass
class ATOMAttentionMetadata:
    """Attention metadata for prefill and decode batched together."""

    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    min_seqlen_q: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    dropout_p: float = 0.0

    max_q_len: Optional[int] = None
    kv_indptr: Optional[torch.Tensor] = None
    kv_indices: Optional[torch.Tensor] = None
    kv_last_page_lens: Optional[torch.Tensor] = None
    cu_seqlen_ks: Optional[torch.Tensor] = None
    cu_seqlen_ke: Optional[torch.Tensor] = None
    sparse_kv_indptr: Optional[torch.Tensor] = None

    work_meta_data: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None

    context: Optional[Context] = None

    def __init__(
        self,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        min_seqlen_q: int = 0,
        slot_mapping: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        max_q_len: Optional[int] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_last_page_lens: Optional[torch.Tensor] = None,
        cu_seqlen_ks: Optional[torch.Tensor] = None,
        cu_seqlen_ke: Optional[torch.Tensor] = None,
        sparse_kv_indptr: Optional[torch.Tensor] = None,
        work_meta_data: Optional[torch.Tensor] = None,
        work_indptr: Optional[torch.Tensor] = None,
        work_info_set: Optional[torch.Tensor] = None,
        reduce_indptr: Optional[torch.Tensor] = None,
        reduce_final_map: Optional[torch.Tensor] = None,
        reduce_partial_map: Optional[torch.Tensor] = None,
        context: Optional[Context] = None,
    ):
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.min_seqlen_q = min_seqlen_q
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.dropout_p = dropout_p
        self.max_q_len = max_q_len
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.kv_last_page_lens = kv_last_page_lens
        self.cu_seqlen_ks = cu_seqlen_ks
        self.cu_seqlen_ke = cu_seqlen_ke
        self.sparse_kv_indptr = sparse_kv_indptr
        self.work_meta_data = work_meta_data
        self.work_indptr = work_indptr
        self.work_info_set = work_info_set
        self.reduce_indptr = reduce_indptr
        self.reduce_final_map = reduce_final_map
        self.reduce_partial_map = reduce_partial_map
        self.context = context

    def asdict_zerocopy(self, skip_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in skip_fields
        }
