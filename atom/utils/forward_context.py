from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Set, Dict, Union
from atom.config import Config, KVCacheTensor
import torch
from abc import ABC, abstractmethod


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
class AttentionMetaData:
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


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[int, Any] = field(default_factory=dict)

    attn_metadata: Optional[
        Union["AttentionMetaData", dict[str, "AttentionMetaData"]]
    ] = None

    kv_cache_data: dict[str, KVCacheTensor] = None

    context: Optional[Context] = None

    def __post_init__(self):
        if not hasattr(self, "no_compile_layers") or self.no_compile_layers is None:
            self.no_compile_layers = {}
        if self.attn_metadata is None:
            self.attn_metadata = {}


_forward_context: Optional[ForwardContext] = ForwardContext()
_forward_kv_cache_context: Optional[ForwardContext] = ForwardContext()


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def set_forward_context(
    attn_metadata: AttentionMetaData, atom_config: Config, context: Context
) -> None:

    global _forward_context
    _forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        no_compile_layers=atom_config.compilation_config.static_forward_context,
        kv_cache_data=_forward_kv_cache_context.kv_cache_data,
        context=context,
    )
    # _forward_context.attn_metadata = attn_metadata
    # _forward_context.no_compile_layers = atom_config.compilation_config.static_forward_context
    # _forward_context = ForwardContext(no_compile_layers=atom_config.compilation_config.static_forward_context, attn_metadata=attn_metadata)


def reset_forward_context() -> None:
    global _forward_context
    _forward_context = ForwardContext()


def set_kv_cache_data(kv_cache_data: dict[int, KVCacheTensor]) -> None:
    global _forward_kv_cache_context
    _forward_kv_cache_context.kv_cache_data = kv_cache_data
