# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional
# from atom.config import Config, KVCacheTensor
import torch
from atom.config import ParallelConfig, config_from_vllm

from vllm.config import CUDAGraphMode
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm import forward_context
from vllm.forward_context import BatchDescriptor
from vllm.config import VllmConfig

from atom.utils.attn_metadata import ATOMAttentionMetadata

def _compute_chunked_local_num_tokens(num_tokens_across_dp_cpu: list[int],
                                      max_num_tokens: int,
                                      chunk_idx: int) -> list[int]:
    dp_size = len(num_tokens_across_dp_cpu)

    local_size = [-1] * dp_size
    for i in range(dp_size):
        dp_tokens = num_tokens_across_dp_cpu[i]
        local_size[i] = min(max_num_tokens,
                            dp_tokens - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # ensure lockstep even if done
    return local_size


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    cu_tokens_across_dp_cpu: torch.Tensor
    local_sizes: Optional[list[int]] = None

    @staticmethod
    def num_tokens_across_dp(num_tokens: int, dp_size: int,
                             dp_rank: int) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from aiter.dist.parallel_state import get_dp_group
        import torch.distributed as dist
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def make(
            parallel_config: ParallelConfig,
            # attn_metadata: Any,
            num_tokens: int,
            num_tokens_across_dp: Optional[torch.Tensor] = None
    ) -> "DPMetadata":

        assert parallel_config.data_parallel_size > 1
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert (num_tokens_across_dp is None
                or num_tokens_across_dp[dp_rank] == batchsize)
        if num_tokens_across_dp is None:
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                batchsize, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
        return DPMetadata(max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu)

    @contextmanager
    def chunked_sizes(self, max_chunk_size_per_rank: int, chunk_idx: int):
        """
        Context manager to compute and temporarily set the per-rank local token
        sizes for a specific chunk during chunked forward execution.
        This is necessary to ensure each DP (data parallel) rank processes its
        designated portion of tokens in lockstep with others, even when the
        token counts are uneven or some ranks have completed their input early.
        For chunked execution, we break up the total tokens on each rank into
        multiple chunks (of at most `max_chunk_size_per_rank`), and for a given
        `chunk_idx`, this context manager sets `self.local_sizes` to the number
        of tokens to process in that chunk on each rank.
        It uses cumulative sizes (`cu_tokens_across_dp_cpu`) to derive the
        number of tokens per rank, and calls `_compute_chunked_local_num_tokens`
        to determine the chunk-wise split.
        `self.local_sizes` is only valid inside the context.
        Args:
            max_chunk_size_per_rank: The max number of tokens each rank is 
                                     allowed to process in this chunk.
            chunk_idx: The index of the chunk to compute sizes for.
        """
        cu_sizes = self.cu_tokens_across_dp_cpu
        num_tokens_across_dp_cpu = [
            (cu_sizes[i] -
             cu_sizes[i - 1]).item() if i > 0 else cu_sizes[0].item()
            for i in range(len(cu_sizes))
        ]
        self.local_sizes = _compute_chunked_local_num_tokens(
            num_tokens_across_dp_cpu, max_chunk_size_per_rank, chunk_idx)
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> Optional[list[int]]:
        return self.local_sizes


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[int, Any]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each 
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    attn_metadata: dict[str, "ATOMAttentionMetadata"] | list[dict[str, "ATOMAttentionMetadata"]]
    virtual_engine: int  # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    ubatch_slices: UBatchSlices | None = None

    def __post_init__(self):
        if not hasattr(self, "no_compile_layers") or self.no_compile_layers is None:
            self.no_compile_layers = {}


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
):
    print('[zeun] ATOM, set_forward_context', flush=True)
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    dp_metadata: Optional[DPMetadata] = None
    atom_config = config_from_vllm(vllm_config)
    if atom_config.parallel_config.data_parallel_size > 1 and num_tokens is not None:
        dp_metadata = DPMetadata.make(atom_config.parallel_config,
                                      # attn_metadata,
                                      num_tokens or 0,
                                      num_tokens_across_dp)

    _forward_context = create_forward_context(
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        virtual_engine=virtual_engine,
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )

    try:
        with override_forward_context(_forward_context):
            yield
    finally:
        pass


def reset_forward_context() -> None:
    global _forward_context
    _forward_context = ForwardContext()


def is_forward_context_available() -> bool:
    return _forward_context is not None


def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
):
    print('[zeun] ATOM, create_forward_context', flush=True)
    atom_config = config_from_vllm(vllm_config)
    return ForwardContext(
        no_compile_layers=atom_config.compilation_config.static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    print('[zeun] ATOM, override_forward_context', flush=True)
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


# monkey patch the vllm forward context method
print('[zeun] ATOM, monkey patch the vllm forward context method', flush=True)
forward_context.get_forward_context = get_forward_context
forward_context.is_forward_context_available = is_forward_context_available
forward_context.create_forward_context = create_forward_context
forward_context.override_forward_context = override_forward_context
forward_context.set_forward_context = set_forward_context
print('[zejun] ATOM, after monkey patch, forward_context.get_forward_context = ', forward_context.get_forward_context, flush=True)
print('[zejun] ATOM, after monkey patch, forward_context.is_forward_context_available = ', forward_context.is_forward_context_available, flush=True)
print('[zejun] ATOM, after monkey patch, forward_context.create_forward_context = ', forward_context.create_forward_context, flush=True)
print('[zejun] ATOM, after monkey patch, forward_context.override_forward_context = ', forward_context.override_forward_context, flush=True)
print('[zejun] ATOM, after monkey patch, forward_context.set_forward_context = ', forward_context.set_forward_context, flush=True)
