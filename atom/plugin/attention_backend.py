from atom.plugin.prepare import is_sglang
from aiter.dist.parallel_state import get_tp_group
import aiter
from functools import lru_cache
import torch

import pdb
import sys
from typing import Any

try:
    from aiter import (
        flash_attn_varlen_func,
        mha_batch_prefill_func,
        dtypes,
        get_pa_metadata_info_v1,
        get_pa_metadata_v1,
        pa_persistent_fwd,
    )
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )


@lru_cache()
def is_fp8_fnuz() -> bool:
    # only device 0 is checked, this assumes MI300 platforms are homogeneous
    return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName

if is_fp8_fnuz():
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = 224.0
else:
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """

    def interaction(self, *args: Any, **kwargs: Any) -> None:
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def print_on_first_rank(*args, **kwargs):
    if get_tp_group().is_first_rank:
        print(*args, **kwargs)

if is_sglang():
    from dataclasses import dataclass
    import torch
    import triton
    import triton.language as tl

    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from typing import Optional
    from atom.utils.forward_context import AttentionMetaData, ForwardContext

    @triton.jit
    def create_flashinfer_kv_indices_triton(
        req_to_token_ptr,  # [max_batch, max_context_len]
        req_pool_indices_ptr,
        page_kernel_lens_ptr,
        kv_indptr,
        kv_start_idx,
        kv_indices_ptr,
        req_to_token_ptr_stride: tl.constexpr,
    ):
        BLOCK_SIZE: tl.constexpr = 512
        pid = tl.program_id(axis=0)

        # find the req pool idx, this is for batch to token
        req_pool_index = tl.load(req_pool_indices_ptr + pid)
        kv_indices_offset = tl.load(kv_indptr + pid)

        kv_start = 0
        kv_end = 0
        if kv_start_idx:
            kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
            kv_end = kv_start
        kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

        num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
        for i in range(num_loop):
            # index into req_to_token_ptr needs to be int64
            offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
            mask = offset < kv_end - kv_start
            data = tl.load(
                req_to_token_ptr
                + req_pool_index * req_to_token_ptr_stride
                + kv_start
                + offset,
                mask=mask,
            )
            tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)

    @dataclass
    class ForwardMetadata:
        # kv_indptr and kv_indices are only used in MLA mode, optional for non-MLA mode
        kv_indptr: Optional[torch.Tensor]
        kv_indices: Optional[torch.Tensor]
        qo_indptr: Optional[torch.Tensor]
        kv_last_page_len: Optional[torch.Tensor]
        max_q_len: Optional[int]
        max_kv_len: Optional[int]
        page_table: Optional[torch.Tensor]
        kv_lens: Optional[torch.Tensor]
        # PA metadata for pa_persistent_fwd (only used in decode mode, non-MLA)
        pa_metadata_qo_indptr: Optional[torch.Tensor] = None
        pa_metadata_pages_kv_indptr: Optional[torch.Tensor] = None
        pa_metadata_kv_indices: Optional[torch.Tensor] = None
        pa_metadata_context_lens: Optional[torch.Tensor] = None
        pa_metadata_max_qlen: Optional[int] = None
        pa_metadata_tp_q_head_num: Optional[int] = None
        # Prefill metadata for mha_batch_prefill_func (only used in prefill mode, non-MLA)
        prefill_pages_kv_indptr: Optional[torch.Tensor] = None
        prefill_kv_indices: Optional[torch.Tensor] = None
        prefill_kv_last_page_lens: Optional[torch.Tensor] = None

    class AiterAttnBackendSglplugin(AiterAttnBackend):
        def __init__(
                self,
                model_runner: ModelRunner,
                skip_prefill: bool = False,
                kv_indptr_buf: Optional[torch.Tensor] = None,
            ):
            super().__init__(model_runner, skip_prefill, kv_indptr_buf)
            print(f"[DEBUG] initializing AiterAttnBackendSglplugin, max_context_len = {self.max_context_len}, page_size = {self.page_size}, model_runner.req_to_token_pool.size = {model_runner.req_to_token_pool.size}", flush=True)
            self.page_table = torch.zeros(
                (model_runner.req_to_token_pool.size, self.max_context_len // self.page_size), dtype=torch.int32, device=model_runner.device
            )

        def init_forward_metadata(self, forward_batch: ForwardBatch):
            """Init auxiliary variables for triton attention backend."""

            bs = forward_batch.batch_size
            kv_indptr = self.kv_indptr
            spec_info = forward_batch.spec_info
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None
            page_table = None

            if forward_batch.forward_mode.is_decode_or_idle():
                if spec_info is None:
                    kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    kv_indices = torch.empty(
                        forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
                    )
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                    bs = kv_indptr.shape[0] - 1

                if self.use_mla:
                    qo_indptr = self.qo_indptr_[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                    kv_last_page_len = self.kv_last_page_len[:bs]
                    max_q_len = 1
                    page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :]

                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        kv_last_page_len,
                        max_q_len,
                        None,
                        page_table,
                        forward_batch.seq_lens,
                    )
                else:
                    # Non-MLA decode mode: use same logic as CUDA Graph mode for page_table construction
                    seq_lens_cpu = forward_batch.seq_lens_cpu
                    if seq_lens_cpu is None:
                        seq_lens_cpu = forward_batch.seq_lens.cpu()
                    
                    # Common setup consistent with CUDA Graph mode (init_forward_metadata_replay_cuda_graph)
                    page_table_persistent = self.page_table
                    seq_lens_persistent = self.seq_lens
                    seq_lens_persistent.fill_(0)
                    page_table_persistent.fill_(0)
                    seq_lens_persistent[:bs].copy_(forward_batch.seq_lens, non_blocking=True)
                    max_seq_pages = (seq_lens_cpu.max().item() + self.page_size - 1) // self.page_size + 1
                    page_table = self.req_to_token[forward_batch.req_pool_indices[:, None], self.strided_indices[:max_seq_pages][None, :],]
                    page_table_persistent[:bs, :max_seq_pages].copy_(page_table // self.page_size, non_blocking=True)
                    
                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        None,  # qo_indptr not used in non-MLA mode
                        None,  # kv_last_page_len not used in non-MLA mode
                        1,     # max_q_len = 1 for decode mode
                        None,
                        page_table_persistent[:bs, :max_seq_pages],
                        seq_lens_persistent[:bs],
                    )
                    
                    # Build pa_metadata for pa_persistent_fwd
                    self._build_pa_metadata_for_decode(bs, tp_q_head_num=self.num_head)
                    return  # Early return for non-MLA decode mode   
            elif forward_batch.forward_mode.is_draft_extend():
                if self.use_mla:
                    kv_indices, kv_indptr, qo_indptr, _ = (
                        spec_info.generate_attn_arg_prefill(
                            forward_batch.req_pool_indices,
                            forward_batch.seq_lens,
                            forward_batch.seq_lens_sum,
                            self.req_to_token,
                        )
                    )
                    self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    None,
                    None,
                )
                else:
                    self.indices_updater_prefill.update(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        prefix_lens=None,
                        encoder_lens=forward_batch.encoder_lens,
                        spec_info=forward_batch.spec_info,
                    )
                    self.forward_metadata = ForwardMetadata(
                        self.indices_updater_prefill.kv_indptr,
                        self.indices_updater_prefill.kv_indices,
                        None,
                        None,
                        self.indices_updater_prefill.max_q_len,
                        self.indices_updater_prefill.max_kv_len,
                        None,
                        None,
                    )
            elif forward_batch.forward_mode.is_target_verify():
                if self.use_mla:
                    draft_num = spec_info.draft_token_num
                    kv_lens = forward_batch.seq_lens + draft_num
                    kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                    device = forward_batch.seq_lens.device

                    qo_indptr = torch.arange(
                        0,
                        (1 + bs) * draft_num,
                        step=draft_num,
                        dtype=torch.int32,
                        device=device,
                    )
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    kv_indices = torch.empty(
                        kv_lens_sum,
                        dtype=torch.int32,
                        device=device,
                    )
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        kv_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        self.kv_last_page_len[:bs],
                        draft_num,
                        None,
                        None,
                        None,
                    )
                else:
                    self.indices_updater_prefill.update(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        prefix_lens=None,
                        encoder_lens=forward_batch.encoder_lens,
                        spec_info=forward_batch.spec_info,
                    )
                    self.forward_metadata = ForwardMetadata(
                        self.indices_updater_prefill.kv_indptr,
                        self.indices_updater_prefill.kv_indices,
                        None,
                        None,
                        self.indices_updater_prefill.max_q_len,
                        self.indices_updater_prefill.max_kv_len,
                        None,
                        None,
                    )
            else:
                prefix_lens = forward_batch.extend_prefix_lens
                prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

                if self.use_mla:
                    self.mla_indices_updater_prefill.update(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        forward_batch.extend_seq_lens,
                        forward_batch.extend_seq_lens.max().item(),
                        forward_batch.seq_lens.max().item(),
                        spec_info=None,
                    )

                    kv_indices = self.mla_indices_updater_prefill.kv_indices

                    self.forward_metadata = ForwardMetadata(
                        self.mla_indices_updater_prefill.kv_indptr,
                        kv_indices,
                        self.mla_indices_updater_prefill.qo_indptr,
                        self.kv_last_page_len[:bs],
                        self.mla_indices_updater_prefill.max_q_len,
                        self.mla_indices_updater_prefill.max_kv_len,
                        None,
                        None,
                    )
                else:
                    self.indices_updater_prefill.update(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        prefix_lens,
                        prefix_lens_cpu=prefix_lens_cpu,
                        seq_lens_cpu=forward_batch.seq_lens_cpu,
                        encoder_lens=forward_batch.encoder_lens,
                        spec_info=None,
                    )
                    # Get page_table for mha_batch_prefill_func
                    page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :]
                    self.forward_metadata = ForwardMetadata(
                        self.indices_updater_prefill.kv_indptr,
                        self.indices_updater_prefill.kv_indices,
                        self.qo_indptr[: bs + 1],  # qo_indptr is set by indices_updater_prefill
                        None,
                        self.indices_updater_prefill.max_q_len,
                        self.indices_updater_prefill.max_kv_len,
                        page_table,
                        forward_batch.seq_lens,
                    )

            if (forward_batch.forward_mode.is_extend() and
                not self.use_mla and
                self.forward_metadata.page_table is not None):
                if self.page_size > 1:
                    seq_lens_cpu = forward_batch.seq_lens_cpu
                    if seq_lens_cpu is None:
                        seq_lens_cpu = forward_batch.seq_lens.cpu()
                    max_seq_pages = (seq_lens_cpu.max().item() + self.page_size - 1) // self.page_size + 1
                    self.forward_metadata.page_table = (
                        self.forward_metadata.page_table[:, self.strided_indices[:max_seq_pages]] // self.page_size
                    )
                self._build_pa_metadata_for_prefill(forward_batch.batch_size)

        def _allocate_pa_metadata_buffers(
            self,
            work_metadata_ptrs_size,
            work_metadata_ptrs_type,
            work_indptr_size,
            work_indptr_type,
            work_info_size,
            work_info_type,
            reduce_indptr_size,
            reduce_indptr_type,
            reduce_final_map_size,
            reduce_final_map_type,
            reduce_partial_map_size,
            reduce_partial_map_type,
        ):
            """Allocate or reuse pa_metadata buffers."""
            if self.pa_metadata_buffers is None:
                self.pa_metadata_buffers = {}
            
            def _get_size_val(size):
                return size[0] if isinstance(size, tuple) else size
            
            # Allocate work_metadata_ptrs
            size_val = _get_size_val(work_metadata_ptrs_size)
            if ("work_metadata_ptrs" not in self.pa_metadata_buffers or 
                self.pa_metadata_buffers["work_metadata_ptrs"].shape[0] < size_val):
                self.pa_metadata_buffers["work_metadata_ptrs"] = torch.empty(
                    work_metadata_ptrs_size, dtype=work_metadata_ptrs_type, device=self.device
                )
            
            # Allocate work_indptr
            size_val = _get_size_val(work_indptr_size)
            if ("work_indptr" not in self.pa_metadata_buffers or 
                self.pa_metadata_buffers["work_indptr"].shape[0] < size_val):
                self.pa_metadata_buffers["work_indptr"] = torch.zeros(
                    work_indptr_size, dtype=work_indptr_type, device=self.device
                )
            else:
                self.pa_metadata_buffers["work_indptr"].zero_()
            
            # Allocate work_info
            size_val = _get_size_val(work_info_size)
            if ("work_info" not in self.pa_metadata_buffers or 
                len(self.pa_metadata_buffers["work_info"].shape) < len(work_info_size) or
                self.pa_metadata_buffers["work_info"].shape[0] < size_val):
                self.pa_metadata_buffers["work_info"] = torch.zeros(
                    work_info_size, dtype=work_info_type, device=self.device
                )
            else:
                self.pa_metadata_buffers["work_info"].zero_()
            
            # Allocate reduce_indptr
            size_val = _get_size_val(reduce_indptr_size)
            if ("reduce_indptr" not in self.pa_metadata_buffers or 
                self.pa_metadata_buffers["reduce_indptr"].shape[0] < size_val):
                self.pa_metadata_buffers["reduce_indptr"] = torch.zeros(
                    reduce_indptr_size, dtype=reduce_indptr_type, device=self.device
                )
            else:
                self.pa_metadata_buffers["reduce_indptr"].zero_()
            
            # Allocate reduce_final_map
            size_val = _get_size_val(reduce_final_map_size)
            if ("reduce_final_map" not in self.pa_metadata_buffers or 
                len(self.pa_metadata_buffers["reduce_final_map"].shape) < len(reduce_final_map_size) or
                self.pa_metadata_buffers["reduce_final_map"].shape[0] < size_val):
                self.pa_metadata_buffers["reduce_final_map"] = torch.zeros(
                    reduce_final_map_size, dtype=reduce_final_map_type, device=self.device
                )
            else:
                self.pa_metadata_buffers["reduce_final_map"].zero_()
            
            # Allocate reduce_partial_map
            reduce_partial_map_size_val = reduce_partial_map_size if isinstance(reduce_partial_map_size, int) else reduce_partial_map_size[0]
            if ("reduce_partial_map" not in self.pa_metadata_buffers or 
                self.pa_metadata_buffers["reduce_partial_map"].shape[0] < reduce_partial_map_size_val):
                self.pa_metadata_buffers["reduce_partial_map"] = torch.zeros(
                    reduce_partial_map_size, dtype=reduce_partial_map_type, device=self.device
                )
            else:
                self.pa_metadata_buffers["reduce_partial_map"].zero_()

        def _build_pa_metadata_for_decode(
            self, 
            batch_size: int, 
            tp_q_head_num: Optional[int] = None,
        ):
            """Build pa_metadata buffers for pa_persistent_fwd in decode mode.
            
            This method prepares all metadata buffers needed for pa_persistent_fwd kernel.
            The metadata can be reused across multiple layers in the same forward pass.
            
            Args:
                batch_size: Batch size for the current forward pass
                tp_q_head_num: Number of Q heads per TP rank. If None, uses self.num_head.
            """
            max_qlen = 1
            
            # Use provided tp_q_head_num or default to self.num_head
            if tp_q_head_num is None:
                tp_q_head_num = self.num_head
            
            kv_dtype_for_metadata = dtypes.fp8
            (
                (work_metadata_ptrs_size, work_metadata_ptrs_type),
                (work_indptr_size, work_indptr_type),
                (work_info_size, work_info_type),
                (reduce_indptr_size, reduce_indptr_type),
                (reduce_final_map_size, reduce_final_map_type),
                (reduce_partial_map_size, reduce_partial_map_type),
            ) = get_pa_metadata_info_v1(
                batch_size,
                max_qlen,
                tp_q_head_num,
                self.q_dtype,
                kv_dtype_for_metadata,
                is_sparse=0,  # 0 for non-sparse attention
                fast_mode=True,
            )
            # Allocate metadata buffers with reuse optimization for multi-layer forward passes
            self._allocate_pa_metadata_buffers(
                work_metadata_ptrs_size,
                work_metadata_ptrs_type,
                work_indptr_size,
                work_indptr_type,
                work_info_size,
                work_info_type,
                reduce_indptr_size,
                reduce_indptr_type,
                reduce_final_map_size,
                reduce_final_map_type,
                reduce_partial_map_size,
                reduce_partial_map_type,
            )
            qo_indptr = self.pa_decode_qo_indptr[: batch_size + 1]
            
            # Get context_lens (kv_lens is always set before calling _build_pa_metadata_for_decode)
            # Note: kv_lens comes from self.seq_lens which is already int32
            context_lens = self.forward_metadata.kv_lens
            
            kernel_block_size = self.page_size
            num_blocks_per_seq = (context_lens + kernel_block_size - 1) // kernel_block_size
            # Use dedicated pa_kv_indptr buffer (similar to self.kv_indptr, but for pa_persistent_fwd)
            pages_kv_indptr = self.pa_kv_indptr[: batch_size + 1]
            pages_kv_indptr[1 : batch_size + 1] = torch.cumsum(num_blocks_per_seq, dim=0)
            
            # Convert page_table to kv_indices (block indices) using Triton kernel to avoid sync
            # page_table shape: [batch_size, max_num_blocks_per_seq]
            # Note: page_table comes from self.page_table which is already int32 and always set before this call
            page_table = self.forward_metadata.page_table
            
            # Use Triton kernel to gather kv_indices from page_table (avoids high-level indexing sync)
            create_flashinfer_kv_indices_triton[(batch_size,)](
                page_table,
                self.pa_batch_indices[:batch_size],  # [0, 1, 2, ..., batch_size-1]
                num_blocks_per_seq,
                pages_kv_indptr,
                None,  # kv_start_idx
                self.pa_kv_indices,
                page_table.stride(0),
            )
            # Use the full buffer - pa_persistent_fwd reads only valid elements based on pages_kv_indptr
            kv_indices = self.pa_kv_indices

            get_pa_metadata_v1(
                seqlens_qo_indptr=qo_indptr,
                pages_kv_indptr=pages_kv_indptr,
                context_lens=context_lens.int(),
                num_heads_per_head_k=tp_q_head_num // self.num_kv_head,
                num_heads_k=self.num_kv_head,
                is_causal=True,
                work_metadata_ptrs=self.pa_metadata_buffers["work_metadata_ptrs"],
                work_indptr=self.pa_metadata_buffers["work_indptr"],
                work_info=self.pa_metadata_buffers["work_info"],
                reduce_indptr=self.pa_metadata_buffers["reduce_indptr"],
                reduce_final_map=self.pa_metadata_buffers["reduce_final_map"],
                reduce_partial_map=self.pa_metadata_buffers["reduce_partial_map"],
                kv_granularity=max(kernel_block_size, 16),
                block_size=kernel_block_size,
                max_seqlen_qo=max_qlen,
                uni_seqlen_qo=max_qlen,
                fast_mode=True,
                topk=-1,
                max_split_per_batch=-1,
            )
            # Store computed values in ForwardMetadata for reuse in forward_decode
            self.forward_metadata.pa_metadata_qo_indptr = qo_indptr
            self.forward_metadata.pa_metadata_pages_kv_indptr = pages_kv_indptr
            self.forward_metadata.pa_metadata_kv_indices = kv_indices
            self.forward_metadata.pa_metadata_context_lens = context_lens
            self.forward_metadata.pa_metadata_max_qlen = max_qlen
            self.forward_metadata.pa_metadata_tp_q_head_num = tp_q_head_num

        def _build_pa_metadata_for_prefill(self, batch_size: int):
            """Build metadata for mha_batch_prefill_func in prefill mode.

            This method prepares page-level metadata needed for mha_batch_prefill_func.
            The metadata is computed once per forward pass and reused across all layers.
            """
            block_size = self.page_size
            context_lens = self.forward_metadata.kv_lens
            num_blocks_per_seq = (context_lens + block_size - 1) // block_size

            # Page-level kv_indptr (reuse pa_kv_indptr buffer)
            pages_kv_indptr = self.pa_kv_indptr[: batch_size + 1]
            pages_kv_indptr[1 : batch_size + 1] = torch.cumsum(num_blocks_per_seq, dim=0)
            
            # Build kv_indices from page_table using triton kernel
            page_table = self.forward_metadata.page_table
            create_flashinfer_kv_indices_triton[(batch_size,)](
                page_table,
                self.pa_batch_indices[:batch_size],
                num_blocks_per_seq,
                pages_kv_indptr,
                None,  # kv_start_idx
                self.pa_kv_indices,
                page_table.stride(0),
            )
            kv_indices = self.pa_kv_indices

            # Compute kv_last_page_lens for each sequence
            kv_last_page_lens = ((context_lens - 1) % block_size + 1).int()

            # Store in ForwardMetadata for reuse in forward_extend
            self.forward_metadata.prefill_pages_kv_indptr = pages_kv_indptr
            self.forward_metadata.prefill_kv_indices = kv_indices
            self.forward_metadata.prefill_kv_last_page_lens = kv_last_page_lens

        def init_cuda_graph_state(
            self,
            max_bs: int,
            max_num_tokens: int,
            kv_indices_buf: Optional[torch.Tensor] = None,
        ):
            self.cuda_graph_kv_last_page_len = torch.ones(max_bs, dtype=torch.int)
            if kv_indices_buf is None:
                self.cuda_graph_kv_indices = torch.zeros(
                    (max_bs * self.max_context_len),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                self.cuda_graph_kv_indices = kv_indices_buf

            # Always use preshuffle layout for pa_fwd_asm
            self.page_table = torch.zeros(
                (max_bs, self.max_context_len // self.page_size), dtype=torch.int32, device=self.device
            )
            self.seq_lens = torch.zeros(
                (max_bs,), dtype=torch.int32, device=self.device
            )
            self.strided_indices = torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            )
            
            # Pre-allocate buffers for pa_metadata in CUDA graph mode (non-MLA decode)
            if not self.use_mla:
                # Pre-allocate pa_metadata buffers for CUDA graph compatibility
                # These buffers will be reused in capture and replay phases
                # Use max_bs and max_qlen=1 (decode mode) to calculate buffer sizes
                max_qlen = 1  # decode mode
                kv_dtype_for_metadata = dtypes.fp8
                (
                    (work_metadata_ptrs_size, work_metadata_ptrs_type),
                    (work_indptr_size, work_indptr_type),
                    (work_info_size, work_info_type),
                    (reduce_indptr_size, reduce_indptr_type),
                    (reduce_final_map_size, reduce_final_map_type),
                    (reduce_partial_map_size, reduce_partial_map_type),
                ) = get_pa_metadata_info_v1(
                    max_bs,
                    max_qlen,
                    self.num_head,  # Use self.num_head as default tp_q_head_num
                    self.q_dtype,
                    kv_dtype_for_metadata,
                    is_sparse=0,
                    fast_mode=True,
                )
                
                # Pre-allocate buffers with maximum size for CUDA graph compatibility
                self._allocate_pa_metadata_buffers(
                    work_metadata_ptrs_size,
                    work_metadata_ptrs_type,
                    work_indptr_size,
                    work_indptr_type,
                    work_info_size,
                    work_info_type,
                    reduce_indptr_size,
                    reduce_indptr_type,
                    reduce_final_map_size,
                    reduce_final_map_type,
                    reduce_partial_map_size,
                    reduce_partial_map_type,
                )

        def init_forward_metadata_capture_cuda_graph(
            self,
            bs: int,
            num_tokens: int,
            req_pool_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            encoder_lens: Optional[torch.Tensor],
            forward_mode: ForwardMode,
            spec_info: Optional[SpecInput],
        ):
            if forward_mode.is_decode_or_idle():
                if self.use_mla:
                    # MLA mode: kv_indptr and kv_indices are used in forward_decode
                    kv_indptr = self.kv_indptr
                    kv_indices = self.cuda_graph_kv_indices
                    if spec_info is None:
                        kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                        kv_indptr = kv_indptr[: bs + 1]
                        create_flashinfer_kv_indices_triton[(bs,)](
                            self.req_to_token,
                            req_pool_indices,
                            seq_lens,
                            kv_indptr,
                            None,
                            kv_indices,
                            self.req_to_token.stride(0),
                        )
                    else:
                        kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

                    qo_indptr = self.qo_indptr_[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )
                    kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                    max_q_len = 1
                    
                    page_table = self.page_table[:bs, :]
                    self.seq_lens[:bs].copy_(seq_lens, non_blocking=True)
                    seq_lens = self.seq_lens[:bs]
                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        kv_last_page_len,
                        max_q_len,
                        None,
                        page_table,
                        seq_lens,
                    )
                else:
                    # Non-MLA decode mode: kv_indptr and kv_indices are NOT used in forward_decode
                    # (forward_decode uses pa_metadata_pages_kv_indptr and pa_metadata_kv_indices instead)
                    page_table = self.page_table[:bs, :]
                    self.seq_lens[:bs].copy_(seq_lens, non_blocking=True)
                    seq_lens_persistent = self.seq_lens[:bs]
                    self.forward_metadata = ForwardMetadata(
                        None,  # kv_indptr not used in non-MLA decode mode
                        None,  # kv_indices not used in non-MLA decode mode
                        None,  # qo_indptr will be set by _build_pa_metadata_for_decode
                        None,  # kv_last_page_len not used in non-MLA mode
                        1,  # max_q_len = 1 for decode mode
                        None,  # max_kv_len
                        page_table,
                        seq_lens_persistent,
                    )
                    
                    # Build pa_metadata using CUDA graph buffers
                    self._build_pa_metadata_for_decode(bs, tp_q_head_num=self.num_head)
                    return  # Early return for non-MLA decode mode

            elif forward_mode.is_target_verify():
                if self.use_mla:
                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[: bs + 1] = torch.arange(
                        0,
                        (1 + bs) * self.num_draft_tokens,
                        step=self.num_draft_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    kv_indptr = self.kv_indptr[: bs + 1]
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indices = self.cuda_graph_kv_indices
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices,
                        seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                    kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                    max_q_len = self.num_draft_tokens

                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        kv_last_page_len,
                        max_q_len,
                        None,
                    )
                else:
                    seq_lens_sum = seq_lens.sum().item()
                    self.indices_updater_prefill.update(
                        req_pool_indices,
                        seq_lens,
                        seq_lens_sum,
                        prefix_lens=None,
                        encoder_lens=encoder_lens,
                        spec_info=spec_info,
                    )
                    self.forward_metadata = ForwardMetadata(
                        self.indices_updater_prefill.kv_indptr,
                        self.indices_updater_prefill.kv_indices,
                        None,
                        None,
                        self.indices_updater_prefill.max_q_len,
                        self.indices_updater_prefill.max_kv_len,
                    )
            elif forward_mode.is_draft_extend():
                num_tokens_per_bs = self.speculative_num_steps + 1
                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    bs * num_tokens_per_bs + 1,
                    step=num_tokens_per_bs,
                    dtype=torch.int32,
                    device=self.device,
                )
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                )
            else:
                raise ValueError(f"Invalid mode: {forward_mode=}")

        def init_forward_metadata_replay_cuda_graph(
            self,
            bs: int,
            req_pool_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            seq_lens_sum: int,
            encoder_lens: Optional[torch.Tensor],
            forward_mode: ForwardMode,
            spec_info: Optional[SpecInput],
            seq_lens_cpu: Optional[torch.Tensor],
            out_cache_loc: Optional[torch.Tensor] = None,
        ):
            if forward_mode.is_decode_or_idle():
                # Common setup for both MLA and non-MLA modes
                page_table_persistent = self.page_table
                seq_lens_persistent = self.seq_lens
                seq_lens_persistent.fill_(0)
                page_table_persistent.fill_(0)
                seq_lens_persistent[:bs].copy_(seq_lens, non_blocking=True)
                max_seq_pages = (seq_lens_cpu.max().item() + self.page_size - 1) // self.page_size + 1
                page_table = self.req_to_token[req_pool_indices[:, None], self.strided_indices[:max_seq_pages][None, :],]
                page_table_persistent[:bs, :max_seq_pages].copy_(page_table // self.page_size, non_blocking=True)
                
                if self.use_mla:
                    # MLA mode: kv_indptr and kv_indices are used in forward_decode
                    kv_indptr = self.kv_indptr
                    kv_indices = self.cuda_graph_kv_indices
                    if spec_info is None:
                        kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                        kv_indptr = kv_indptr[: bs + 1]
                        create_flashinfer_kv_indices_triton[(bs,)](
                            self.req_to_token,
                            req_pool_indices[:bs],
                            seq_lens[:bs],
                            kv_indptr,
                            None,
                            kv_indices,
                            self.req_to_token.stride(0),
                        )
                    else:
                        kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                        kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
                    
                    qo_indptr = self.qo_indptr_[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )
                    kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                    max_q_len = 1
                    
                    self.forward_metadata = ForwardMetadata(
                        kv_indptr,
                        kv_indices,
                        qo_indptr,
                        kv_last_page_len,
                        max_q_len,
                        None,
                        page_table_persistent[:bs, :max_seq_pages],
                        seq_lens_persistent[:bs],
                    )
                else:
                    # Non-MLA decode mode: kv_indptr and kv_indices are NOT used in forward_decode
                    # (forward_decode uses pa_metadata_pages_kv_indptr and pa_metadata_kv_indices instead)
                    self.forward_metadata = ForwardMetadata(
                        None,  # kv_indptr not used in non-MLA decode mode
                        None,  # kv_indices not used in non-MLA decode mode
                        None,  
                        None,  # kv_last_page_len not used in non-MLA mode
                        1,  # max_q_len = 1 for decode mode
                        None,  # max_kv_len
                        page_table_persistent[:bs, :max_seq_pages],
                        seq_lens_persistent[:bs],
                    )
                    
                    # Rebuild pa_metadata using CUDA graph buffers (updates content, keeps same addresses)
                    self._build_pa_metadata_for_decode(bs, tp_q_head_num=self.num_head)

            elif forward_mode.is_target_verify():
                bs = len(req_pool_indices)
                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * self.num_draft_tokens,
                    step=self.num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                )
                kv_lens = seq_lens + self.num_draft_tokens
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            elif forward_mode.is_draft_extend():
                seq_lens = seq_lens[:bs]
                accept_lens = spec_info.accept_length[:bs]
                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                raise ValueError("Invalid forward mode")

        def get_cuda_graph_seq_len_fill_value(self):
            return 1

        def forward_extend(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            layer: RadixAttention,
            forward_batch: ForwardBatch,
            save_kv_cache=True,
        ):
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )

            self.logits_soft_cap = layer.logit_cap

            if save_kv_cache:
                assert k is not None
                assert v is not None
                if self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
                    # Shuffle operation is already fused in rotary_emb, so just save directly
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

            if self.use_mla:
                max_q_len = self.forward_metadata.max_q_len
                max_kv_len = self.forward_metadata.max_kv_len
                kv_indptr = self.forward_metadata.kv_indptr
                kv_indices = self.forward_metadata.kv_indices
                qo_indptr = self.forward_metadata.qo_indptr
                K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
                kv_lora_rank = V_Buffer.shape[-1]
                qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
                qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
                assert len(q.shape) == 3
                assert len(k.shape) == 3
                assert len(v.shape) == 3

                if (
                    forward_batch.forward_mode.is_extend()
                    and not forward_batch.forward_mode.is_target_verify()
                    and not forward_batch.forward_mode.is_draft_extend()
                ):
                    if kv_indices.shape[0] == 0:
                        o = flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            qo_indptr,
                            max_q_len,
                            max_q_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )
                        return o
                    elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                        K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                        kvc, k_pe = torch.split(
                            K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                        )
                        kvprefix = layer.kv_b_proj(kvc.contiguous())[0]

                        kvprefix = kvprefix.view(
                            -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                        )
                        k_prefix, v_prefix = torch.split(
                            kvprefix, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                        )
                        k_prefix = torch.cat(
                            [
                                k_prefix,
                                torch.broadcast_to(
                                    k_pe,
                                    (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                                ),
                            ],
                            dim=-1,
                        )
                        assert (
                            forward_batch.extend_prefix_lens.shape
                            == forward_batch.extend_seq_lens.shape
                        )

                        k = k_prefix
                        v = v_prefix

                        o = flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr,
                            max_q_len,
                            max_kv_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )
                        return o

                    else:
                        if layer.qk_head_dim != layer.v_head_dim:
                            o = q.new_empty(
                                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                            )
                        else:
                            o = torch.empty_like(q)

                        mla_prefill_fwd(
                            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                            K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                            qo_indptr,
                            kv_indptr,
                            kv_indices,
                            self.forward_metadata.kv_last_page_len,
                            self.forward_metadata.max_q_len,
                            layer.scaling,
                            layer.logit_cap,
                        )
                        K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                        return o
                elif forward_batch.forward_mode.is_target_verify():
                    o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                    mla_decode_fwd(
                        q,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o,
                        self.forward_metadata.qo_indptr,
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                    return o
                elif forward_batch.forward_mode.is_draft_extend():
                    o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                    kv_indptr = self.forward_metadata.kv_indptr
                    kv_indices = self.forward_metadata.kv_indices
                    mla_prefill_fwd(
                        q,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o,
                        self.forward_metadata.qo_indptr,
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                    return o
                else:
                    raise ValueError(
                        f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                    )
            else:
                # Non-MLA prefill: use mha_batch_prefill_func with paged KV cache (same layout as pa_persistent_fwd in decode)
                batch_size = forward_batch.batch_size
                bs0 = batch_size + 1

                # Get paged KV cache from token_to_kv_pool (already saved via set_kv_buffer)
                k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                num_slots, num_kv_heads, head_size = k_buffer.shape
                block_size = self.page_size
                num_blocks = num_slots // block_size
                
                k_buffer = k_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)
                v_buffer = v_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)

                # KV cache is already in 5D vectorized format (shuffled in fused rotary_emb)
                # Just view_as to the correct shape (no permute needed)
                # x is the vectorization factor: 16 bytes (128-bit) / dtype size
                quant_dtype = dtypes.fp8
                x = 16 // quant_dtype.itemsize  # 128-bit vector alignment
                k_cache_template = torch.empty(
                    [num_blocks, num_kv_heads, head_size // x, block_size, x],
                    dtype=k_buffer.dtype,
                    device="meta",
                )
                v_cache_template = torch.empty(
                    [num_blocks, num_kv_heads, block_size // x, head_size, x],
                    dtype=v_buffer.dtype,
                    device="meta",
                )
                new_key_cache = k_buffer.view_as(k_cache_template)
                new_value_cache = v_buffer.view_as(v_cache_template)

                q_fp8 = q.to(dtypes.fp8)
                pages_kv_indptr = self.forward_metadata.prefill_pages_kv_indptr
                kv_indices = self.forward_metadata.prefill_kv_indices
                kv_last_page_lens = self.forward_metadata.prefill_kv_last_page_lens

                o = mha_batch_prefill_func(
                    q=q_fp8,
                    k=new_key_cache,
                    v=new_value_cache,
                    cu_seqlens_q=self.qo_indptr[:bs0],
                    kv_indptr=pages_kv_indptr,
                    kv_page_indices=kv_indices,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=self.forward_metadata.max_kv_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    q_descale=self.q_scale,
                    k_descale=self.k_scale,
                    v_descale=self.v_scale,
                    kv_last_page_lens=kv_last_page_lens,
                )

                return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        def forward_decode(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            layer: RadixAttention,
            forward_batch: ForwardBatch,
            save_kv_cache=True,
        ):
            q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

            # Create o as 3D tensor [batch_size, num_heads, head_dim] for both MLA and pa_fwd_asm
            # In decode mode, q.shape[0] equals batch_size (each sequence has 1 token)
            # Use q.shape[0] instead of forward_batch.batch_size to be safe
            batch_size = q.shape[0]
            head_dim_out = layer.v_head_dim if layer.qk_head_dim != layer.v_head_dim else layer.head_dim
            o = q.new_empty((batch_size, layer.tp_q_head_num, head_dim_out))

            if save_kv_cache:
                if self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, k, v
                    )
                else:
                    # Shuffle operation is already fused in rotary_emb, so just save directly
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

            if self.use_mla:
                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                mla_decode_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    layer.scaling,
                    layer.logit_cap,
                )
                k_buffer = k_buffer.view(-1, 1, layer.qk_head_dim)
            else:
                k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                num_slots, num_kv_heads, head_size = k_buffer.shape
                block_size = self.page_size
                num_blocks = num_slots // block_size
                k_buffer = k_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)
                v_buffer = v_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)


                quant_dtype = dtypes.fp8
                x = 16 // quant_dtype.itemsize
                k_cache_template = torch.empty(
                    [num_blocks, num_kv_heads, head_size // x, block_size, x],
                    dtype=k_buffer.dtype,
                    device="meta",
                )
                # V: [num_blocks, block_size, num_kv_heads, head_size] -> [num_blocks, num_kv_heads, block_size // x, head_size, x]
                v_cache_template = torch.empty(
                    [num_blocks, num_kv_heads, block_size // x, head_size, x],
                    dtype=v_buffer.dtype,
                    device="meta",
                )
                new_key_cache = k_buffer.view_as(k_cache_template)
                new_value_cache = v_buffer.view_as(v_cache_template)
                
                total_tokens = num_blocks * block_size
                k_qscale = self.k_qscale[:, :total_tokens]
                v_qscale = self.v_qscale[:, :total_tokens]
                
                q = q.view(batch_size, layer.tp_q_head_num, layer.head_dim)
        

                assert self.forward_metadata.pa_metadata_qo_indptr is not None, "pa_metadata_qo_indptr should be set by _build_pa_metadata_for_decode"
                assert self.forward_metadata.pa_metadata_pages_kv_indptr is not None, "pa_metadata_pages_kv_indptr should be set by _build_pa_metadata_for_decode"
                assert self.forward_metadata.pa_metadata_kv_indices is not None, "pa_metadata_kv_indices should be set by _build_pa_metadata_for_decode"
                assert self.forward_metadata.pa_metadata_context_lens is not None, "pa_metadata_context_lens should be set by _build_pa_metadata_for_decode"
                assert self.forward_metadata.pa_metadata_max_qlen is not None, "pa_metadata_max_qlen should be set by _build_pa_metadata_for_decode"
                
                qo_indptr = self.forward_metadata.pa_metadata_qo_indptr
                kv_indptr = self.forward_metadata.pa_metadata_pages_kv_indptr
                kv_indices = self.forward_metadata.pa_metadata_kv_indices
                context_lens = self.forward_metadata.pa_metadata_context_lens
                max_qlen = self.forward_metadata.pa_metadata_max_qlen
                
                
                _, _ = pa_persistent_fwd(
                    Q=q,
                    K=new_key_cache,
                    V=new_value_cache,
                    output=o,
                    max_qlen=max_qlen,
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    context_lens=context_lens,
                    work_indptr=self.pa_metadata_buffers["work_indptr"],
                    work_info=self.pa_metadata_buffers["work_info"],
                    reduce_indptr=self.pa_metadata_buffers["reduce_indptr"],
                    reduce_final_map=self.pa_metadata_buffers["reduce_final_map"],
                    reduce_partial_map=self.pa_metadata_buffers["reduce_partial_map"],
                    K_QScale=k_qscale,
                    V_QScale=v_qscale,
                    softmax_scale=layer.scaling,
                    mask=1,  
                )
            return o.view(-1, layer.tp_q_head_num * head_dim_out)
    

        # def init_forward_metadata(self, forward_batch: ForwardBatch):
        #     """Init auxiliary variables for triton attention backend."""
        #     print_on_first_rank(f"[DEBUG] call init_forward_metadata from atom!!!", flush=True)
        #     bs = forward_batch.batch_size
        #     kv_indptr = self.kv_indptr
        #     spec_info = forward_batch.spec_info
        #     qo_indptr = None
        #     kv_last_page_len = None
        #     max_q_len = None

        #     work_metadata = None
        #     work_indptr = None
        #     work_info_set = None
        #     reduce_indptr = None
        #     reduce_final_map = None
        #     reduce_partial_map = None

        #     num_kv_splits = None
        #     # num_kv_splits_indptr = None
        #     print(f"[DEBUG] forward_batch.req_pool_indices shape = {forward_batch.req_pool_indices.shape}, forward_batch.req_to_token_pool.req_to_token shape = {forward_batch.req_to_token_pool.req_to_token.shape}", flush=True)
        #     page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :]
        #     if self.page_size > 1:
        #         strided_indices = torch.arange(
        #             0, page_table.shape[1], self.page_size, device=self.device
        #         )

        #         page_table = (
        #             page_table[:, strided_indices] // self.page_size
        #         )

        #     if forward_batch.forward_mode.is_decode_or_idle():
        #         print_on_first_rank(f"[DEBUG] forward mode is decode or idle!!!", flush=True)
        #         if spec_info is None:
        #             print_on_first_rank(f"[DEBUG] spec_info is None, need to create kv indices for batch size {bs}!!!, forward_batch.seq_lens = {forward_batch.seq_lens}, forward_batch.seq_lens_sum={forward_batch.seq_lens_sum}", flush=True)
        #             kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
        #             kv_indptr = kv_indptr[: bs + 1]
        #             kv_indices = torch.empty(
        #                 forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
        #             )
        #             print(f"============ before decode self.req_to_token = {self.req_to_token[0, :30]}", flush=True)
        #             create_flashinfer_kv_indices_triton[(bs,)](
        #                 self.req_to_token,
        #                 forward_batch.req_pool_indices,
        #                 forward_batch.seq_lens,
        #                 kv_indptr,
        #                 None,
        #                 kv_indices,
        #                 self.req_to_token.stride(0),
        #             )
        #             print(f"============ after decode self.req_to_token = {self.req_to_token[0, :30]}", flush=True)
        #         else:
        #             print_on_first_rank(f"[DEBUG] spec_info is provided, use kv indices from spec_info!!!", flush=True)
        #             kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
        #             bs = kv_indptr.shape[0] - 1

        #         assert not self.use_mla, "mla is not supported for now."

        #         self.forward_metadata = ForwardMetadata(
        #             kv_indptr,
        #             kv_indices,
        #             qo_indptr,
        #             kv_last_page_len,
        #             max_q_len,
        #             None,
        #             page_table=page_table,
        #             work_metadata=work_metadata,
        #             work_info_set=work_info_set,
        #             work_indptr=work_indptr,
        #             reduce_indptr=reduce_indptr,
        #             reduce_final_map=reduce_final_map,
        #             reduce_partial_map=reduce_partial_map,
        #             num_kv_splits=num_kv_splits,
        #             run_graph=False,
        #         )
        #     else:
        #         print_on_first_rank(f"[DEBUG] forward mode is prefill!!!", flush=True)
        #         prefix_lens = forward_batch.extend_prefix_lens
        #         assert not self.use_mla, "mla is not supported for now."

        #         self.indices_updater_prefill.update(
        #             forward_batch.req_pool_indices,
        #             forward_batch.seq_lens,
        #             forward_batch.seq_lens_sum,
        #             prefix_lens,
        #             encoder_lens=forward_batch.encoder_lens,
        #             spec_info=None,
        #         )
        #         self.forward_metadata = ForwardMetadata(
        #             self.indices_updater_prefill.kv_indptr,
        #             self.indices_updater_prefill.kv_indices,
        #             None,
        #             None,
        #             self.indices_updater_prefill.max_q_len,
        #             self.indices_updater_prefill.max_kv_len,
        #             page_table=page_table,
        #         )

        # def init_forward_metadata_capture_cuda_graph(
        #     self,
        #     bs: int,
        #     num_tokens: int,
        #     req_pool_indices: torch.Tensor,
        #     seq_lens: torch.Tensor,
        #     encoder_lens: Optional[torch.Tensor],
        #     forward_mode: ForwardMode,
        #     spec_info: Optional[SpecInput],
        # ):

        #     num_kv_splits = None
        #     # num_kv_splits_indptr = None

        #     work_metadata = None
        #     work_info_set = None
        #     work_indptr = None

        #     reduce_indptr = None
        #     reduce_final_map = None
        #     reduce_partial_map = None

        #     if forward_mode.is_decode_or_idle():
        #         qo_indptr = None
        #         kv_last_page_len = None
        #         max_q_len = None

        #         if spec_info is None:
        #             kv_indptr = self.kv_indptr
        #             kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
        #             kv_indptr = kv_indptr[: bs + 1]
        #             kv_indices = self.cuda_graph_kv_indices
        #             create_flashinfer_kv_indices_triton[(bs,)](
        #                 self.req_to_token,
        #                 req_pool_indices,
        #                 seq_lens,
        #                 kv_indptr,
        #                 None,
        #                 kv_indices,
        #                 self.req_to_token.stride(0),
        #             )
        #         else:
        #             kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

        #         assert not self.use_mla, "mla is not supported for now."

        #         self.forward_metadata = ForwardMetadata(
        #             kv_indptr,
        #             kv_indices,
        #             qo_indptr,
        #             kv_last_page_len,
        #             max_q_len,
        #             kv_indptr[-1].item(),
        #             work_metadata=work_metadata,
        #             work_info_set=work_info_set,
        #             work_indptr=work_indptr,
        #             reduce_indptr=reduce_indptr,
        #             reduce_final_map=reduce_final_map,
        #             reduce_partial_map=reduce_partial_map,
        #             num_kv_splits=num_kv_splits,
        #             # num_kv_splits_indptr=num_kv_splits_indptr,
        #         )

        #     else:
        #         raise ValueError(f"Invalid mode: {forward_mode=}")

        # def init_forward_metadata_replay_cuda_graph(
        #     self,
        #     bs: int,
        #     req_pool_indices: torch.Tensor,
        #     seq_lens: torch.Tensor,
        #     seq_lens_sum: int,
        #     encoder_lens: Optional[torch.Tensor],
        #     forward_mode: ForwardMode,
        #     spec_info: Optional[SpecInput],
        #     seq_lens_cpu: Optional[torch.Tensor],
        # ):

        #     if forward_mode.is_decode_or_idle():
        #         kv_indptr = self.kv_indptr
        #         kv_indices = self.cuda_graph_kv_indices
        #         if spec_info is None:
        #             kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
        #             kv_indptr = kv_indptr[: bs + 1]
        #             create_flashinfer_kv_indices_triton[(bs,)](
        #                 self.req_to_token,
        #                 req_pool_indices[:bs],
        #                 seq_lens[:bs],
        #                 kv_indptr,
        #                 None,
        #                 kv_indices,
        #                 self.req_to_token.stride(0),
        #             )
        #         else:
        #             kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
        #             kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
        #     else:
        #         raise ValueError("Invalid forward mode")


        # def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        #     print_on_first_rank(f"[DEBUG] call forward_decode from atom!!!", flush=True)
        #     return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)
        #     q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        #     if layer.qk_head_dim != layer.v_head_dim:
        #         o = q.new_empty(
        #             (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
        #             dtype=self.input_dtype,
        #         )
        #     else:
        #         o = torch.empty_like(q, dtype=self.input_dtype)

        #     if save_kv_cache:
        #         forward_batch.token_to_kv_pool.set_kv_buffer(
        #             layer, forward_batch.out_cache_loc, k, v
        #         )

        #     self.logits_soft_cap = layer.logit_cap

        #     k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
        #         layer.layer_id
        #     )
        #     num_slots, num_kv_heads, head_size = k_buffer.shape
        #     block_size = self.page_size
        #     num_blocks = num_slots // block_size
        #     k_buffer = k_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)
        #     v_buffer = v_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)

        #     x = 16 // k_buffer.element_size()
        #     k_cache_template = torch.empty(
        #         [num_blocks, num_kv_heads, head_size // x, block_size, x],
        #         dtype=k_buffer.dtype,
        #         device="meta",
        #     )
        #     # V: [num_blocks, block_size, num_kv_heads, head_size] -> [num_blocks, num_kv_heads, block_size // x, head_size, x]
        #     v_cache_template = torch.empty(
        #         [num_blocks, num_kv_heads, head_size, block_size],
        #         dtype=v_buffer.dtype,
        #         device="meta",
        #     )
        #     k_cache = k_buffer.view_as(k_cache_template)
        #     v_cache = v_buffer.view_as(v_cache_template)

        #     # TODO kkhuang-amd need to remove it when paged_attention_ragged support fp8-kv
        #     # if self.kv_cache_dtype == fp8_dtype:
        #     #     dtype = q.dtype

        #     #     k_cache = k_cache.to(dtype)
        #     #     v_cache = v_cache.to(dtype)

        #     if get_tp_group().is_first_rank and layer.layer_id == 0:
        #         print(f"[DEBUG] forward_decode: o shape = {o.shape}, q shape = {q.shape}, k_cache shape = {k_cache.shape}, v_cache shape = {v_cache.shape}, k_cache contiguous = {k_cache.is_contiguous()}, v_cache contiguous = {v_cache.is_contiguous()}", flush=True)
        #         print(f"k_scale shape = {self.k_scale.shape}, v_scale shape = {self.v_scale.shape}", flush=True)
        #         print(f"page_table shape = {self.forward_metadata.page_table.shape}, forward_batch.seq_lens shape = {forward_batch.seq_lens.shape}, page_table.stride(0) = {self.forward_metadata.page_table.stride(0)}", flush=True)
        #         print(f"page_table = {self.forward_metadata.page_table}, seq_lens = {forward_batch.seq_lens}", flush=True)
        #         print(f"self.k_scale = {self.k_scale}, self.v_scale = {self.v_scale}", flush=True)

        #     page_table = self.forward_metadata.page_table
        #     o = aiter.pa_fwd_asm(
        #         q,
        #         k_cache,
        #         v_cache,
        #         page_table,
        #         forward_batch.seq_lens,
        #         page_table.stride(0),
        #         K_QScale=None,
        #         V_QScale=None,
        #         out_=None,
        #         high_precision=0,
        #     )

        #     return o.view(-1, layer.tp_q_head_num * layer.head_dim)
            # return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)

        def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            print_on_first_rank(f"[DEBUG] call forward_extend from atom!!!", flush=True)
            # ForkedPdb().set_trace()
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )

            self.logits_soft_cap = layer.logit_cap

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    if self.use_mla:
                        forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                    else:
                        forward_batch.token_to_kv_pool.set_kv_buffer(
                            layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                        )

            if get_tp_group().is_first_rank and layer.layer_id == 0:
                print(f"[DEBUG] forward_prefill: q shape = {q.shape}, k shape = {k.shape}, v shape = {v.shape}, layer.tp_q_head_num = {layer.tp_q_head_num}, layer.head_dim = {layer.head_dim}", flush=True)
                print(f"self.qo_indptr shape = {self.qo_indptr.shape}, kv_indptr shape = {self.forward_metadata.kv_indptr.shape}, kv_indices shape = {self.forward_metadata.kv_indices.shape}, max_q_len = {self.forward_metadata.max_q_len}, max_kv_len = {self.forward_metadata.max_kv_len}, logits_soft_cap = {self.logits_soft_cap}", flush=True)

            seqlens_in_batch = forward_batch.seq_lens
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            o = aiter.flash_attn_varlen_func(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim),
                v.contiguous().view(-1, layer.tp_v_head_num, layer.head_dim),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=self.forward_metadata.max_q_len,
                max_seqlen_k=self.forward_metadata.max_kv_len,
                min_seqlen_q=0,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=True,
                window_size=(-1, -1, 0),
                sink_ptr=None,
            )

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)
            # return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)