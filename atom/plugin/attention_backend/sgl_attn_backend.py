from __future__ import annotations

"""
end to end attention solution with aiter kernels
"""

from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

import sglang.srt.layers.attention.aiter_backend as _sglang_aiter
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_bool_env_var
from atom.model_ops.attention_mla import (
    _MLA_MIN_HEADS,
    triton_convert_req_index_to_global_index,
    triton_convert_req_index_to_global_index_dsa_prefill,
)

logger = logging.getLogger("atom")

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_func,
        dtypes,
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
        get_pa_metadata_info_v1,
        get_pa_metadata_v1,
        pa_fwd_asm,
        pa_persistent_fwd,
        mla_decode_fwd,
    )
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

# MLA prefill kernels - imported separately to avoid breaking the main aiter imports
mla_prefill_ps_asm_fwd = None
mla_reduce_v1 = None
mla_prefill_fwd = None
try:
    from aiter import mla_prefill_ps_asm_fwd
except ImportError:
    pass
try:
    from aiter import mla_reduce_v1
except ImportError:
    pass
try:
    from aiter.mla import mla_prefill_fwd
    from aiter.mla import mla_decode_fwd
except ImportError:
    pass

import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_shuffle_kernel(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
    slot_mapping_ptr,  # [num_tokens]
    k_scale_ptr,
    v_scale_ptr,
    x,
    k_stride0,
    v_stride0,
    block_size,
    head_size,
    num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
    QUANT: tl.constexpr,
):
    tid = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    src_offset_k = tid * k_stride0 + head_id * head_size
    src_offset_v = tid * v_stride0 + head_id * head_size
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    dst_offset = (
        block_id * num_kv_heads * head_size * block_size
        + head_id * head_size * block_size
    )
    dst_k_shuffle_offset = (
        dst_offset + offset // x * block_size * x + block_offset * x + offset % x
    )
    dst_v_shuffle_offset = (
        dst_offset + block_offset // x * head_size * x + offset * x + block_offset % x
    )
    k_val = tl.load(key_ptr + src_offset_k + offset)
    v_val = tl.load(value_ptr + src_offset_v + offset)
    if QUANT:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        k_dtype = key_cache_ptr.type.element_ty
        v_dtype = value_cache_ptr.type.element_ty
        k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
        v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)
    tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
    tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)


def reshape_and_cache_shuffle_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
):
    num_tokens = slot_mapping.shape[0]
    _, num_kv_heads, head_size = key.shape
    num_blocks, block_size, _, _ = key_cache.shape
    x = 16 // key_cache.element_size()
    k_cache_template = torch.empty(
        [num_blocks, num_kv_heads, head_size // x, block_size, x],
        dtype=key_cache.dtype,
        device="meta",
    )
    v_cache_template = torch.empty(
        [num_blocks, num_kv_heads, block_size // x, head_size, x],
        dtype=value_cache.dtype,
        device="meta",
    )
    new_key_cache = key_cache.view_as(k_cache_template)
    new_value_cache = value_cache.view_as(v_cache_template)
    QUANT = False
    if kv_cache_dtype.startswith("fp8"):
        QUANT = True
    grid = (
        num_tokens,
        num_kv_heads,
    )
    reshape_and_cache_shuffle_kernel[grid](
        key,
        value,
        new_key_cache,
        new_value_cache,
        slot_mapping,
        k_scales,
        v_scales,
        x,
        key.stride(0),
        value.stride(0),
        block_size,
        head_size,
        num_kv_heads,
        BLOCK_SIZE=head_size,
        QUANT=QUANT,
    )


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
    # mla
    work_metadata: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None
    fp8_prefill_kv_indices: Optional[torch.Tensor] = None
    num_kv_splits: Optional[int] = None
    # PA metadata for pa_persistent_fwd (only used in decode mode, non-MLA)
    pa_metadata_qo_indptr: Optional[torch.Tensor] = None
    pa_metadata_pages_kv_indptr: Optional[torch.Tensor] = None
    pa_metadata_kv_indices: Optional[torch.Tensor] = None
    pa_metadata_context_lens: Optional[torch.Tensor] = None
    pa_metadata_max_qlen: Optional[int] = None
    pa_metadata_tp_q_head_num: Optional[int] = None
    # Prefill metadata for mha_batch_prefill_func (only used in prefill mode, non-MLA)
    # prefill_pages_kv_indptr: Optional[torch.Tensor] = None
    # prefill_kv_indices: Optional[torch.Tensor] = None
    # prefill_kv_last_page_lens: Optional[torch.Tensor] = None
    


class ATOMAttnBackendForSgl(AiterAttnBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(model_runner, skip_prefill, kv_indptr_buf)
        mapping = getattr(
            model_runner.token_to_kv_pool, "full_attention_layer_id_mapping", None
        )

        if isinstance(mapping, dict) and mapping:
            first_full_attn_id = next(iter(mapping.keys()))
        else:
            first_full_attn_id = 0

        self.q_dtype = model_runner.dtype  # Save q dtype for pa_metadata building
        hf_config = getattr(getattr(model_runner, "model_config", None), "hf_config", None)
        self._atom_sparse_topk_tokens = getattr(hf_config, "index_topk", None)
        self._atom_sparse_topk_fallback_logged = False

        # assert not self.use_mla, "MLA mode is not implemented yet in ATOMAttnBackendForSgl."

        # Pre-initialized qo_indptr for pa_persistent_fwd decode mode: [0, 1, 2, ..., max_bs]
        # In decode mode, each sequence has 1 token, so this is always [0, 1, 2, ..., batch_size]
        max_bs = model_runner.req_to_token_pool.size
        self.pa_decode_qo_indptr = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        self.seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.page_table = torch.zeros(
            (max_bs, self.max_context_len // self.page_size),
            dtype=torch.int32,
            device=model_runner.device,
        )
        # Pre-compute strided indices for page_table construction (used in both CUDA Graph and non-CUDA Graph modes)
        self.strided_indices = torch.arange(
            0, self.max_context_len, self.page_size, device=model_runner.device
        )

        if not self.use_mla:
            # Pre-allocate buffers for pa_persistent_fwd (used in both CUDA graph and non-CUDA graph modes)
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            max_total_blocks = max_bs * max_num_blocks_per_seq
            self.pa_kv_indices = torch.zeros(
                max_total_blocks, dtype=torch.int32, device=self.device
            )
            # Pre-allocate pa_kv_indptr buffer (similar to self.kv_indptr, but dedicated for pa_persistent_fwd)
            self.pa_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            # Pre-initialized batch indices [0, 1, 2, ..., max_bs-1] for Triton kernel
            self.pa_batch_indices = torch.arange(
                0, max_bs, dtype=torch.int32, device=self.device
            )

        # Pre-allocated descale tensors for FP8 attention (q, k, v all use scale=1.0)

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        self.pa_metadata_buffers = None

        k_buffer, _ = model_runner.token_to_kv_pool.get_kv_buffer(first_full_attn_id)
        num_slots, num_kv_heads, _ = k_buffer.shape
        block_size = self.page_size
        num_blocks = num_slots // block_size
        max_total_tokens = num_blocks * block_size
        self.k_qscale = torch.ones(
            num_kv_heads, max_total_tokens, dtype=torch.float32, device=self.device
        )
        self.v_qscale = torch.ones(
            num_kv_heads, max_total_tokens, dtype=torch.float32, device=self.device
        )
        self.decode_using_pa_ps = self.page_size == 1024

        if not self.use_mla:
            self._num_kv_heads = num_kv_heads
            gqa_group_size = self.num_head // num_kv_heads
            padded_group = 1 << (gqa_group_size - 1).bit_length() if gqa_group_size & (gqa_group_size - 1) else gqa_group_size
            self._gqa_group_size = gqa_group_size
            self._padded_gqa_group_size = padded_group
            self._padded_q_heads = padded_group * num_kv_heads
            self._need_q_pad = self._padded_q_heads != self.num_head
            self._mla_kernel_heads = self.num_head
            self._use_mla_ps_kernel = False
        else:
            self._padded_q_heads = self.num_head
            self._need_q_pad = False
            self._mla_kernel_heads = self._get_required_mla_kernel_heads(self.num_head)
            self._use_mla_ps_kernel = (
                _sglang_aiter._use_mla_ps_kernel
                and self._mla_kernel_heads % _MLA_MIN_HEADS == 0
            )
            if self._mla_kernel_heads != self.num_head:
                logger.info(
                    "Pad MLA kernel heads for SGLang plugin: %d -> %d.",
                    self.num_head,
                    self._mla_kernel_heads,
                )

    def _get_required_mla_kernel_heads(self, num_q_heads: int) -> int:
        if num_q_heads >= _MLA_MIN_HEADS and num_q_heads % _MLA_MIN_HEADS == 0:
            return num_q_heads
        return math.lcm(num_q_heads, _MLA_MIN_HEADS)

    def _get_mla_head_repeat_info(self, num_q_heads: int) -> tuple[int, int]:
        padded_q_heads = self._get_required_mla_kernel_heads(num_q_heads)
        repeat_factor = padded_q_heads // num_q_heads
        if repeat_factor == 1:
            return num_q_heads, 1
        if not getattr(self, "_atom_mla_head_repeat_logged", False):
            logger.info(
                "Enable MLA head repeat for SGLang plugin: %d -> %d "
                "(repeat factor %d).",
                num_q_heads,
                padded_q_heads,
                repeat_factor,
            )
            self._atom_mla_head_repeat_logged = True
        return padded_q_heads, repeat_factor

    def _log_sparse_topk_fallback_once(self, mode: str) -> None:
        if self._atom_sparse_topk_fallback_logged:
            return
        logger.info(
            "SGLang plugin synthesized full sparse MLA top-k for %s because "
            "topk_indices was missing.",
            mode,
        )
        self._atom_sparse_topk_fallback_logged = True

    def _build_full_sparse_topk_decode(
        self, seq_lens: torch.Tensor, topk_tokens: int
    ) -> torch.Tensor:
        seq_lens = seq_lens.to(device=seq_lens.device, dtype=torch.int32)
        arange = torch.arange(
            topk_tokens, dtype=torch.int32, device=seq_lens.device
        ).unsqueeze(0)
        topk = arange.expand(seq_lens.shape[0], -1).clone()
        topk[topk >= seq_lens.unsqueeze(1)] = -1
        return topk

    def _maybe_build_sparse_topk_fallback(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        topk_tokens = self._atom_sparse_topk_tokens
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if self.page_size != 1 or topk_tokens is None or seq_lens is None:
            return None
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        seq_lens_for_check = seq_lens_cpu if seq_lens_cpu is not None else seq_lens
        if seq_lens_for_check.numel() == 0:
            return None

        seq_lens = seq_lens.to(dtype=torch.int32)
        if int(seq_lens_for_check.max().item()) > int(topk_tokens):
            return None

        if forward_batch.forward_mode.is_decode_or_idle():
            self._log_sparse_topk_fallback_once("decode")
            return self._build_full_sparse_topk_decode(seq_lens, int(topk_tokens))

        if forward_batch.forward_mode.is_extend_without_speculative():
            # Synthetic top-k for extend was removed: it assumed batch-contiguous
            # global token ids (cumsum(seq_lens)), which does not match radix /
            # paged req_to_token layout and produced invalid physical KV slots.
            return None

        return None

    def _prepare_mla_decode_io(
        self, q: torch.Tensor, layer: RadixAttention
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        padded_q_heads, repeat_factor = self._get_mla_head_repeat_info(
            layer.tp_q_head_num
        )
        if repeat_factor > 1:
            q = q.repeat_interleave(repeat_factor, dim=1)
        o = q.new_empty(
            (q.shape[0], padded_q_heads, layer.v_head_dim),
            dtype=self.input_dtype,
        )
        return q, o, repeat_factor

    def _finalize_mla_decode_output(
        self,
        o: torch.Tensor,
        layer: RadixAttention,
        repeat_factor: int,
        flatten_output: bool,
    ) -> torch.Tensor:
        if repeat_factor > 1:
            o = o[:, ::repeat_factor, :].contiguous()
        if flatten_output:
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return o

    def _run_mla_decode(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        layer: RadixAttention,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        max_q_len: int,
        work_metadata=None,
        work_indptr=None,
        work_info_set=None,
        reduce_indptr=None,
        reduce_final_map=None,
        reduce_partial_map=None,
        num_kv_splits=None,
        flatten_output: bool = True,
    ) -> torch.Tensor:
        q, o, repeat_factor = self._prepare_mla_decode_io(q, layer)
        kv_scale = layer.k_scale if layer.k_scale is not None else self.k_scale
        mla_decode_fwd(
            q,
            k_buffer.view(-1, 1, 1, layer.qk_head_dim),
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            max_q_len,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            work_meta_data=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            q_scale=kv_scale,
            kv_scale=kv_scale,
            intra_batch_mode=_sglang_aiter.intra_batch_mode,
            num_kv_splits=num_kv_splits,
        )
        return self._finalize_mla_decode_output(
            o,
            layer,
            repeat_factor,
            flatten_output=flatten_output,
        )

    def _run_mla_prefill(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        layer: RadixAttention,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        max_q_len: int,
        flatten_output: bool = True,
    ) -> torch.Tensor:
        q, o, repeat_factor = self._prepare_mla_decode_io(q, layer)
        mla_prefill_fwd(
            q,
            k_buffer.view(-1, 1, 1, layer.qk_head_dim),
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            max_q_len,
            layer.scaling,
            layer.logit_cap,
        )
        return self._finalize_mla_decode_output(
            o,
            layer,
            repeat_factor,
            flatten_output=flatten_output,
        )

    def make_mla_decode_meta_data_buffer(self, max_seqlen_qo, batch_size):
        dtype = self.kv_cache_dtype

        if self.enable_dp_attention:
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            self.max_split_per_batch = min(
                (cu_num + batch_size - 1) // batch_size, self.fix_max_split_per_batch
            )

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            batch_size,
            max_seqlen_qo,
            self._mla_kernel_heads,
            dtype,
            dtype,
            is_sparse=False,
            fast_mode=_sglang_aiter.fast_mode,
            num_kv_splits=self.max_split_per_batch,
            intra_batch_mode=_sglang_aiter.intra_batch_mode,
        )

        work_metadata = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device="cuda"
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device="cuda"
        )
        work_info_set = torch.empty(
            work_info_set_size,
            dtype=work_info_set_type,
            device="cuda",
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
        )

        return (
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_meta_data(
        self,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_q_len,
        fast_mode,
        max_split_per_batch,
        intra_batch_mode,
    ):
        get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            self._mla_kernel_heads,
            1,
            False,
            work_metadata,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(self.page_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=fast_mode,
            max_split_per_batch=max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
            dtype_q=self.kv_cache_dtype,
            dtype_kv=self.kv_cache_dtype,
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

                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
                num_kv_splits = None

                if self._use_mla_ps_kernel:
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_q_len, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,  # max_kv_len
                    None,  # page_table
                    None,  # kv_lens
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )

            else:
                if self.decode_using_pa_ps:
                    # Non-MLA decode mode: use same logic as CUDA Graph mode for page_table construction
                    seq_lens_cpu = forward_batch.seq_lens_cpu
                    if seq_lens_cpu is None:
                        seq_lens_cpu = forward_batch.seq_lens.cpu()

                    # Common setup consistent with CUDA Graph mode (init_forward_metadata_replay_cuda_graph)
                    page_table_persistent = self.page_table
                    seq_lens_persistent = self.seq_lens
                    seq_lens_persistent.fill_(0)
                    page_table_persistent.fill_(0)
                    seq_lens_persistent[:bs].copy_(
                        forward_batch.seq_lens, non_blocking=True
                    )
                    max_seq_pages = (
                        seq_lens_cpu.max().item() + self.page_size - 1
                    ) // self.page_size + 1
                    page_table = self.req_to_token[
                        forward_batch.req_pool_indices[:, None],
                        self.strided_indices[:max_seq_pages][None, :],
                    ]
                    page_table_persistent[:bs, :max_seq_pages].copy_(
                        page_table // self.page_size, non_blocking=True
                    )
                else:
                    page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    None,  # qo_indptr not used in non-MLA mode
                    None,  # kv_last_page_len not used in non-MLA mode
                    1,  # max_q_len = 1 for decode mode
                    None,
                    (
                        page_table_persistent[:bs, :max_seq_pages]
                        if self.decode_using_pa_ps
                        else page_table
                    ),
                    (
                        seq_lens_persistent[:bs]
                        if self.decode_using_pa_ps
                        else forward_batch.seq_lens
                    ),
                )

                # Build pa_metadata for pa_persistent_fwd
                if self.decode_using_pa_ps:
                    self._build_pa_metadata_for_decode(bs, tp_q_head_num=self._padded_q_heads)
                # return  # Early return for non-MLA decode mode
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.use_mla:
                # raise NotImplementedError("MLA prefill mode is not implemented yet in ATOMAttnBackendForSgl.")
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    forward_batch.extend_seq_lens.max().item(),
                    forward_batch.seq_lens.max().item(),
                    spec_info=None
                )

                max_q_len = self.mla_indices_updater_prefill.max_q_len
                qo_indptr = self.mla_indices_updater_prefill.qo_indptr

                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                fp8_prefill_kv_indices = None
                reduce_partial_map = None
                
                from sglang.srt.utils import is_gfx95_supported
                _use_fp8_prefill_attn = (
                    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and is_gfx95_supported()
                )
                if _use_fp8_prefill_attn:
                    tile_q = 256
                    qlen_granularity = tile_q // (self.num_head // self.num_kv_head)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map
                    ) = self.make_mla_prefill_ps_meta_data_buffer(
                        bs, max_q_len, qlen_granularity
                    )


                    self.make_mla_prefill_ps_meta_data(
                        qo_indptr,
                        qo_indptr,
                        forward_batch.seq_lens,
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        is_causal=True,
                    )

                    total_s = int(forward_batch.extend_seq_lens.sum())
                    fp8_prefill_kv_indices = torch.arange(
                        total_s, device=self.device, dtype=torch.int32
                    )
                
                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    self.mla_indices_updater_prefill.kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    fp8_prefill_kv_indices=fp8_prefill_kv_indices,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )
                # Get page_table for mha_batch_prefill_func
                page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, :
                ]
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    self.qo_indptr[
                        : bs + 1
                    ],  # qo_indptr is set by indices_updater_prefill
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                    None,
                    forward_batch.seq_lens,
                )

        if (
            forward_batch.forward_mode.is_extend()
            and not self.use_mla
            and self.forward_metadata.page_table is not None
        ):
            if self.page_size > 1:
                seq_lens_cpu = forward_batch.seq_lens_cpu
                if seq_lens_cpu is None:
                    seq_lens_cpu = forward_batch.seq_lens.cpu()
                max_seq_pages = (
                    seq_lens_cpu.max().item() + self.page_size - 1
                ) // self.page_size + 1
                self.forward_metadata.page_table = (
                    self.forward_metadata.page_table[
                        :, self.strided_indices[:max_seq_pages]
                    ]
                    // self.page_size
                )
            if self.decode_using_pa_ps:
                self._build_pa_metadata_for_prefill(forward_batch.batch_size)
        if (
            not self.decode_using_pa_ps
            and self.page_size > 1
            and self.forward_metadata.page_table is not None
        ):
            self.forward_metadata.page_table = (
                self.forward_metadata.page_table[:, self.strided_indices]
                // self.page_size
            )

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
        if (
            "work_metadata_ptrs" not in self.pa_metadata_buffers
            or self.pa_metadata_buffers["work_metadata_ptrs"].shape[0] < size_val
        ):
            self.pa_metadata_buffers["work_metadata_ptrs"] = torch.empty(
                work_metadata_ptrs_size,
                dtype=work_metadata_ptrs_type,
                device=self.device,
            )

        # Allocate work_indptr
        size_val = _get_size_val(work_indptr_size)
        if (
            "work_indptr" not in self.pa_metadata_buffers
            or self.pa_metadata_buffers["work_indptr"].shape[0] < size_val
        ):
            self.pa_metadata_buffers["work_indptr"] = torch.zeros(
                work_indptr_size, dtype=work_indptr_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["work_indptr"].zero_()

        # Allocate work_info
        size_val = _get_size_val(work_info_size)
        if (
            "work_info" not in self.pa_metadata_buffers
            or len(self.pa_metadata_buffers["work_info"].shape) < len(work_info_size)
            or self.pa_metadata_buffers["work_info"].shape[0] < size_val
        ):
            self.pa_metadata_buffers["work_info"] = torch.zeros(
                work_info_size, dtype=work_info_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["work_info"].zero_()

        # Allocate reduce_indptr
        size_val = _get_size_val(reduce_indptr_size)
        if (
            "reduce_indptr" not in self.pa_metadata_buffers
            or self.pa_metadata_buffers["reduce_indptr"].shape[0] < size_val
        ):
            self.pa_metadata_buffers["reduce_indptr"] = torch.zeros(
                reduce_indptr_size, dtype=reduce_indptr_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["reduce_indptr"].zero_()

        # Allocate reduce_final_map
        size_val = _get_size_val(reduce_final_map_size)
        if (
            "reduce_final_map" not in self.pa_metadata_buffers
            or len(self.pa_metadata_buffers["reduce_final_map"].shape)
            < len(reduce_final_map_size)
            or self.pa_metadata_buffers["reduce_final_map"].shape[0] < size_val
        ):
            self.pa_metadata_buffers["reduce_final_map"] = torch.zeros(
                reduce_final_map_size, dtype=reduce_final_map_type, device=self.device
            )
        else:
            self.pa_metadata_buffers["reduce_final_map"].zero_()

        # Allocate reduce_partial_map
        reduce_partial_map_size_val = (
            reduce_partial_map_size
            if isinstance(reduce_partial_map_size, int)
            else reduce_partial_map_size[0]
        )
        if (
            "reduce_partial_map" not in self.pa_metadata_buffers
            or self.pa_metadata_buffers["reduce_partial_map"].shape[0]
            < reduce_partial_map_size_val
        ):
            self.pa_metadata_buffers["reduce_partial_map"] = torch.zeros(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=self.device,
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

        # kv_dtype_for_metadata = dtypes.fp8
        (
            (work_metadata_ptrs_size, work_metadata_ptrs_type),
            (work_indptr_size, work_indptr_type),
            (work_info_size, work_info_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_pa_metadata_info_v1(
            batch_size,
            self.num_kv_head,
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
        # kv_indices = self.pa_kv_indices

        # Compute kv_last_page_lens for each sequence
        # kv_last_page_lens = ((context_lens - 1) % block_size + 1).int()

        # Store in ForwardMetadata for reuse in forward_extend
        # self.forward_metadata.prefill_pages_kv_indptr = pages_kv_indptr
        # self.forward_metadata.prefill_kv_indices = kv_indices
        # self.forward_metadata.prefill_kv_last_page_lens = kv_last_page_lens

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_kv_last_page_len = torch.ones(max_bs, dtype=torch.int, device=self.device)
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
            (max_bs, self.max_context_len // self.page_size),
            dtype=torch.int32,
            device=self.device,
        )
        self.seq_lens = torch.zeros((max_bs,), dtype=torch.int32, device=self.device)
        self.strided_indices = torch.arange(
            0, self.max_context_len, self.page_size, device=self.device
        )
        
        if self.use_mla and self._use_mla_ps_kernel:
            max_seqlen_qo = 1
            (
                self.work_metadata,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, max_bs)
        elif self.use_mla:
            self.work_metadata = None
            self.work_indptr = None
            self.work_info_set = None
            self.reduce_indptr = None
            self.reduce_final_map = None
            self.reduce_partial_map = None

        if self.decode_using_pa_ps and not self.use_mla:
            (
                (work_metadata_ptrs_size, work_metadata_ptrs_type),
                (work_indptr_size, work_indptr_type),
                (work_info_size, work_info_type),
                (reduce_indptr_size, reduce_indptr_type),
                (reduce_final_map_size, reduce_final_map_type),
                (reduce_partial_map_size, reduce_partial_map_type),
            ) = get_pa_metadata_info_v1(
                max_bs,
                self.num_kv_head,
            )
            
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
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
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

                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
                num_kv_splits = None

                if self._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                page_table = self.page_table[:bs, :]
                self.seq_lens[:bs].copy_(seq_lens, non_blocking=True)
                seq_lens_persistent = self.seq_lens[:bs]
                self.forward_metadata = ForwardMetadata(
                    None,
                    None,
                    None,
                    None,
                    1,
                    None,
                    page_table,
                    seq_lens_persistent,
                )
                
                if self.decode_using_pa_ps:
                    self._build_pa_metadata_for_decode(bs, tp_q_head_num=self._padded_q_heads)
                return
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
            if self.use_mla:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
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

                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
                num_kv_splits = None

                if self._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                page_table_persistent = self.page_table
                seq_lens_persistent = self.seq_lens
                seq_lens_persistent.fill_(0)
                page_table_persistent.fill_(0)
                seq_lens_persistent[:bs].copy_(seq_lens, non_blocking=True)
                max_seq_pages = (seq_lens_cpu.max().item() + self.page_size - 1) // self.page_size + 1
                page_table = self.req_to_token[req_pool_indices[:, None], self.strided_indices[:max_seq_pages][None, :],]
                page_table_persistent[:bs, :max_seq_pages].copy_(page_table // self.page_size, non_blocking=True)

                self.forward_metadata = ForwardMetadata(
                    None,
                    None,
                    None,
                    None,
                    1,
                    None,
                    page_table_persistent[:bs, :max_seq_pages],
                    seq_lens_persistent[:bs],
                )
                
                if self.decode_using_pa_ps:
                    self._build_pa_metadata_for_decode(bs, tp_q_head_num=self._padded_q_heads)
        else:
            raise ValueError("Invalid forward mode")

    def set_kv_buffer_with_layout_shuffle(
        self,
        cache_loc,
        k,
        v,
        k_buffer,
        v_buffer,
        k_scale,
        v_scale,
        block_size,
    ):
        num_slots, num_kv_heads, head_dim = k_buffer.shape
        num_blocks = num_slots // block_size
        num_slots_with_block = num_blocks * block_size
        k_buffer = k_buffer[:num_slots_with_block].view(
            num_blocks, block_size, num_kv_heads, head_dim
        )
        v_buffer = v_buffer[:num_slots_with_block].view(
            num_blocks, block_size, num_kv_heads, head_dim
        )
        reshape_and_cache_shuffle_triton(
            k,
            v,
            k_buffer,
            v_buffer,
            cache_loc,
            "auto",
            k_scale,
            v_scale,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
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
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v
                    )
                else:
                    k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    self.set_kv_buffer_with_layout_shuffle(
                        cache_loc, k, v, k_buffer, v_buffer,
                        layer.k_scale, layer.v_scale, self.page_size,
                    )

        if self.use_mla:
            return self._forward_extend_mla(
                q, k, v, layer, forward_batch, topk_indices=topk_indices
            )
        else:
            return self._forward_extend_mha(q, k, v, layer, forward_batch)

    def _forward_extend_mha(self, q, k, v, layer, forward_batch):
        """Non-MLA extend path: standard MHA with flash_attn_varlen_func."""
        seqlens_in_batch = forward_batch.seq_lens
        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        if q.dtype != k.dtype and k.dtype == dtypes.fp8:
            q = q.to(dtypes.fp8)
        o = flash_attn_varlen_func(
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

    def _forward_extend_mla(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """MLA extend path: ported from sglang aiter_backend forward_extend MLA logic."""
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
            if topk_indices is None:
                topk_indices = self._maybe_build_sparse_topk_fallback(forward_batch)
            if topk_indices is not None and topk_indices.shape[0] != q.shape[0]:
                topk_indices = None
            if topk_indices is not None:
                return self._forward_extend_mla_sparse(
                    q, layer, forward_batch, K_Buffer, topk_indices
                )
            return self._forward_extend_mla_normal(
                q, k, v, layer, forward_batch,
                K_Buffer, V_Buffer,
                kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim,
                max_q_len, max_kv_len, kv_indptr, kv_indices, qo_indptr,
            )
        elif forward_batch.forward_mode.is_target_verify():
            return self._forward_extend_mla_target_verify(
                q, layer, K_Buffer, qo_indptr,
            )
        elif forward_batch.forward_mode.is_draft_extend():
            return self._forward_extend_mla_draft_extend(
                q, layer, K_Buffer, qo_indptr,
            )
        else:
            raise ValueError(
                f"Invalid forward mode for MLA extend: {forward_batch.forward_mode=}"
            )

    def _build_sparse_decode_kv_indptr(
        self, seq_lens: torch.Tensor, topk_tokens: int
    ) -> torch.Tensor:
        sparse_kv_indptr = torch.zeros(
            seq_lens.shape[0] + 1,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        sparse_kv_indptr[1:] = torch.cumsum(
            torch.clamp(seq_lens.to(torch.int32), max=topk_tokens),
            dim=0,
        )
        return sparse_kv_indptr

    def _build_sparse_prefill_metadata(
        self, forward_batch: ForwardBatch, topk_tokens: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
        num_tokens = int(extend_seq_lens.sum().item())
        sparse_cu_seqlens_q = torch.arange(
            num_tokens + 1, dtype=torch.int32, device=device
        )
        sparse_kv_indptr = torch.zeros(
            num_tokens + 1, dtype=torch.int32, device=device
        )
        if num_tokens == 0:
            return (
                sparse_cu_seqlens_q,
                sparse_kv_indptr,
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )

        per_token_kv_lens = [
            torch.arange(1, int(count) + 1, dtype=torch.int32, device=device).clamp_(
                max=topk_tokens
            )
            for count in extend_seq_lens.cpu().tolist()
            if count > 0
        ]
        if per_token_kv_lens:
            sparse_kv_indptr[1:] = torch.cumsum(torch.cat(per_token_kv_lens), dim=0)

        token_to_seq_idxs = torch.repeat_interleave(
            torch.arange(
                extend_seq_lens.shape[0], dtype=torch.int32, device=device
            ),
            extend_seq_lens.to(torch.int64),
        )
        sparse_kv_last_page_lens = torch.ones(
            num_tokens, dtype=torch.int32, device=device
        )
        return (
            sparse_cu_seqlens_q,
            sparse_kv_indptr,
            token_to_seq_idxs,
            sparse_kv_last_page_lens,
        )

    def _forward_extend_mla_sparse(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k_buffer: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.page_size == 1, (
            "ATOM sparse MLA bridge for SGLang currently requires page_size == 1."
        )
        topk_indices = topk_indices.to(torch.int32).contiguous()
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        max_seq_len = (
            int(seq_lens_cpu.max().item())
            if seq_lens_cpu is not None
            else int(forward_batch.seq_lens.max().item())
        )
        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices.to(torch.long), :max_seq_len
        ].to(torch.int32)
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        cu_seqlens_k = torch.zeros(
            seq_lens.shape[0] + 1, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k[1:] = torch.cumsum(seq_lens, dim=0)
        (
            sparse_cu_seqlens_q,
            sparse_kv_indptr,
            token_to_seq_idxs,
            sparse_kv_last_page_lens,
        ) = self._build_sparse_prefill_metadata(
            forward_batch, topk_indices.shape[1], q.device
        )
        if sparse_kv_last_page_lens.numel() == 0:
            return q.new_empty(
                (0, layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.input_dtype,
            )

        sparse_kv_indices = triton_convert_req_index_to_global_index_dsa_prefill(
            sparse_cu_seqlens_q,
            sparse_kv_indptr,
            token_to_seq_idxs,
            topk_indices,
            block_tables,
            cu_seqlens_k,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )
        # Padding uses -1; invalid / masked slots may still be -1. Clamp into a valid
        # KV slot range so get_mla_kv_buffer_triton never faults; mla_prefill_fwd only
        # uses keys within sparse_kv_indptr ranges.
        kv_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        max_slot = max(int(kv_buf.shape[0]) - 1, 0)
        gather_loc = sparse_kv_indices.clamp(min=0, max=max_slot)
        sparse_k_nope, sparse_k_rope = forward_batch.token_to_kv_pool.get_mla_kv_buffer(
            layer,
            gather_loc,
            dst_dtype=q.dtype,
        )
        sparse_k_buffer = torch.cat([sparse_k_nope, sparse_k_rope], dim=-1)
        compact_kv_indices = torch.arange(
            sparse_k_buffer.shape[0], dtype=torch.int32, device=q.device
        )
        return self._run_mla_prefill(
            q,
            sparse_k_buffer,
            layer,
            sparse_cu_seqlens_q,
            sparse_kv_indptr,
            compact_kv_indices,
            sparse_kv_last_page_lens,
            1,
            flatten_output=True,
        )

    def _forward_extend_mla_normal(
        self, q, k, v, layer, forward_batch,
        K_Buffer, V_Buffer,
        kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim,
        max_q_len, max_kv_len, kv_indptr, kv_indices, qo_indptr,
    ):
        """Normal MLA extend (not target_verify, not draft_extend).

        Three sub-paths mirroring sglang aiter_backend:
        1) No prefix -> fp8 prefill kernel (mla_prefill_ps_asm_fwd) or flash_attn fallback
        2) Has prefix, absorbed weights differ -> decompress via kv_b_proj + flash_attn
        3) Has prefix, qk_head_dim matches -> mla_prefill_fwd kernel
        """
        extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

        if kv_indices.shape[0] == 0 or extend_no_prefix:
            # --- Sub-path 1: no prefix, pure prefill ---
            use_fp8_prefill = (
                self.forward_metadata.fp8_prefill_kv_indices is not None
            )
            if use_fp8_prefill:
                total_s = q.shape[0]
                nhead = layer.tp_q_head_num
                v_head_dim = layer.v_head_dim

                if q.dtype != dtypes.fp8:
                    q = q.to(dtypes.fp8)
                if k.dtype != dtypes.fp8:
                    k = k.to(dtypes.fp8)
                if v.dtype != dtypes.fp8:
                    v = v.to(dtypes.fp8)
                one_scale = torch.ones(
                    (), dtype=torch.float32, device=q.device
                )

                kv_indptr_asm = qo_indptr
                kv_indices_asm = self.forward_metadata.fp8_prefill_kv_indices

                tile_q = 256
                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                logits = torch.empty(
                    (reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                attn_lse = torch.empty(
                    (reduce_partial_map.size(0) * tile_q, nhead),
                    dtype=torch.float32,
                    device=q.device,
                )
                final_lse = torch.empty(
                    (total_s, nhead),
                    dtype=torch.float32,
                    device=q.device,
                )
                output = q.new_empty(
                    (total_s, nhead, v_head_dim),
                    dtype=self.input_dtype,
                )

                mla_prefill_ps_asm_fwd(
                    q,
                    k,
                    v,
                    qo_indptr,
                    kv_indptr_asm,
                    kv_indices_asm,
                    self.forward_metadata.work_indptr,
                    self.forward_metadata.work_info_set,
                    max_q_len,
                    layer.scaling,
                    True,
                    logits,
                    attn_lse,
                    output,
                    one_scale,
                    one_scale,
                    one_scale,
                )
                mla_reduce_v1(
                    logits,
                    attn_lse,
                    reduce_indptr,
                    reduce_final_map,
                    reduce_partial_map,
                    tile_q,
                    output,
                    final_lse,
                )
            elif layer.qk_head_dim == (kv_lora_rank + qk_rope_head_dim) and mla_prefill_fwd is not None:
                # Absorbed MLA: head_dim (576) exceeds CK limit (256),
                # use mla_prefill_fwd which natively supports large MLA head dims.
                # For no-prefix, use input k (bfloat16) directly instead of K_Buffer
                # (which may be FP8). mla_prefill_fwd doesn't support FP8 KV.
                total_s = q.shape[0]
                temp_kv_indices = torch.arange(
                    total_s, device=q.device, dtype=torch.int32
                )
                output = self._run_mla_prefill(
                    q,
                    k,
                    layer,
                    qo_indptr,
                    qo_indptr,
                    temp_kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    max_q_len,
                )
            else:
                output = flash_attn_varlen_func(
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
            return output

        elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
            # --- Sub-path 2: has prefix, need kv_b_proj decompress ---
            K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
            kvc, k_pe = torch.split(
                K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
            )

            if self.kv_cache_dtype == dtypes.fp8:
                dtype = q.dtype
                kvc = kvc.to(dtype)
                k_pe = k_pe.to(dtype)

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

            o = flash_attn_varlen_func(
                q,
                k_prefix,
                v_prefix,
                qo_indptr,
                kv_indptr,
                max_q_len,
                max_kv_len,
                softmax_scale=layer.scaling,
                causal=True,
            )
            return o

        else:
            # --- Sub-path 3: has prefix, qk_head_dim == kv_lora_rank + qk_rope_head_dim ---
            # Gather needed KV entries and cast to bf16 (K_Buffer may be FP8)
            k_selected = torch.index_select(K_Buffer, 0, kv_indices)
            if k_selected.dtype != q.dtype:
                k_selected = k_selected.to(q.dtype)
            compact_kv_indices = torch.arange(
                k_selected.shape[0], device=q.device, dtype=torch.int32
            )

            if layer.qk_head_dim != layer.v_head_dim:
                o = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                )
            else:
                o = torch.empty_like(q)

            o = self._run_mla_prefill(
                q,
                k_selected,
                layer,
                qo_indptr,
                kv_indptr,
                compact_kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
            )
            return o

    def _forward_extend_mla_target_verify(self, q, layer, K_Buffer, qo_indptr):
        """MLA target_verify path (speculative decoding verification)."""
        work_metadata = self.forward_metadata.work_metadata
        work_indptr = self.forward_metadata.work_indptr
        work_info_set = self.forward_metadata.work_info_set
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map
        num_kv_splits = self.forward_metadata.num_kv_splits

        return self._run_mla_decode(
            q,
            K_Buffer,
            layer,
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.forward_metadata.kv_last_page_len,
            self.forward_metadata.max_q_len,
            work_metadata=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
            flatten_output=False,
        )

    def _forward_extend_mla_draft_extend(self, q, layer, K_Buffer, qo_indptr):
        """MLA draft_extend path (speculative decoding draft extension)."""
        work_metadata = self.forward_metadata.work_metadata
        work_indptr = self.forward_metadata.work_indptr
        work_info_set = self.forward_metadata.work_info_set
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map
        num_kv_splits = self.forward_metadata.num_kv_splits

        return self._run_mla_decode(
            q,
            K_Buffer,
            layer,
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.forward_metadata.kv_last_page_len,
            self.forward_metadata.max_q_len,
            work_metadata=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
            flatten_output=False,
        )


    def forward_decode_pa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            self.set_kv_buffer_with_layout_shuffle(
                forward_batch.out_cache_loc,
                k,
                v,
                k_buffer,
                v_buffer,
                layer.k_scale,
                layer.v_scale,
                self.page_size,
            )

        if self.use_mla:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            work_metadata = self.forward_metadata.work_metadata
            work_indptr = self.forward_metadata.work_indptr
            work_info_set = self.forward_metadata.work_info_set

            reduce_indptr = self.forward_metadata.reduce_indptr
            reduce_final_map = self.forward_metadata.reduce_final_map
            reduce_partial_map = self.forward_metadata.reduce_partial_map

            num_kv_splits = self.forward_metadata.num_kv_splits

            mla_decode_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.forward_metadata.qo_indptr,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=work_metadata,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=layer.k_scale,
                kv_scale=layer.k_scale,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
            
        else:
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            block_size = self.page_size
            num_slots, num_kv_heads, head_size = k_buffer.shape
            num_blocks = num_slots // block_size
            k_buffer = k_buffer[: num_blocks * block_size].view(
                num_blocks, block_size, num_kv_heads, head_size
            )
            v_buffer = v_buffer[: num_blocks * block_size].view(
                num_blocks, block_size, num_kv_heads, head_size
            )

            x = 16 // k_buffer.element_size()
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
            q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            pa_fwd_asm(
                Q=q,
                K=new_key_cache,
                V=new_value_cache,
                block_tables=self.forward_metadata.page_table,
                context_lens=self.forward_metadata.kv_lens,
                block_tables_stride0=self.forward_metadata.page_table.stride(0),
                K_QScale=self.k_scale,
                V_QScale=self.v_scale,
                out_=o,
            )
            return o

    def forward_decode_pa_ps(
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
        head_dim_out = (
            layer.v_head_dim
            if layer.qk_head_dim != layer.v_head_dim
            else layer.head_dim
        )
        o = q.new_empty((batch_size, layer.tp_q_head_num, head_dim_out))

        if save_kv_cache:
            if self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            else:
                k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
                self.set_kv_buffer_with_layout_shuffle(forward_batch.out_cache_loc, k, v, k_buffer, v_buffer, layer.k_scale, layer.v_scale, self.page_size)

        if self.use_mla:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            work_metadata = self.forward_metadata.work_metadata
            work_indptr = self.forward_metadata.work_indptr
            work_info_set = self.forward_metadata.work_info_set

            reduce_indptr = self.forward_metadata.reduce_indptr
            reduce_final_map = self.forward_metadata.reduce_final_map
            reduce_partial_map = self.forward_metadata.reduce_partial_map

            num_kv_splits = self.forward_metadata.num_kv_splits

            mla_decode_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.forward_metadata.qo_indptr,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=work_metadata,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=layer.k_scale,
                kv_scale=layer.k_scale,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
        else:
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            num_slots, num_kv_heads, head_size = k_buffer.shape
            block_size = self.page_size
            num_blocks = num_slots // block_size
            k_buffer = k_buffer[: num_blocks * block_size].view(
                num_blocks, block_size, num_kv_heads, head_size
            )
            v_buffer = v_buffer[: num_blocks * block_size].view(
                num_blocks, block_size, num_kv_heads, head_size
            )

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

            if self._need_q_pad:
                nkv = self._num_kv_heads
                gs = self._gqa_group_size
                pgs = self._padded_gqa_group_size
                q = q.view(batch_size, nkv, gs, layer.head_dim)
                q = torch.nn.functional.pad(q, (0, 0, 0, pgs - gs))
                q = q.reshape(batch_size, self._padded_q_heads, layer.head_dim)
                o = q.new_empty((batch_size, self._padded_q_heads, head_dim_out))

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

            if self._need_q_pad:
                nkv = self._num_kv_heads
                gs = self._gqa_group_size
                pgs = self._padded_gqa_group_size
                o = o.view(batch_size, nkv, pgs, head_dim_out)
                o = o[:, :, :gs, :].contiguous()
                o = o.view(batch_size, layer.tp_q_head_num, head_dim_out)

        return o.view(-1, layer.tp_q_head_num * head_dim_out)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.use_mla:
            return self._forward_decode_mla(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                topk_indices=topk_indices,
            )
        else:
            if self.decode_using_pa_ps:
                return self.forward_decode_pa_ps(
                    q, k, v, layer, forward_batch, save_kv_cache
                )
            else:
                return self.forward_decode_pa(q, k, v, layer, forward_batch, save_kv_cache)

    def _forward_decode_mla_sparse(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        k_buffer: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.page_size == 1, (
            "ATOM sparse MLA bridge for SGLang currently requires page_size == 1."
        )
        topk_indices = topk_indices.to(torch.int32).contiguous()
        sparse_kv_indptr = self._build_sparse_decode_kv_indptr(
            self.forward_metadata.kv_indptr[1:] - self.forward_metadata.kv_indptr[:-1],
            topk_indices.shape[1],
        )
        sparse_kv_last_page_lens = torch.ones(
            sparse_kv_indptr.shape[0] - 1, dtype=torch.int32, device=q.device
        )
        sparse_kv_indices = triton_convert_req_index_to_global_index(
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            sparse_kv_indptr,
            self.forward_metadata.kv_indices,
            topk_indices,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None
        num_kv_splits = None
        if self.work_metadata is not None and self._use_mla_ps_kernel:
            self.make_mla_meta_data(
                self.forward_metadata.qo_indptr,
                sparse_kv_indptr,
                sparse_kv_last_page_lens,
                self.work_metadata,
                self.work_info_set,
                self.work_indptr,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
                self.forward_metadata.max_q_len,
                fast_mode=_sglang_aiter.fast_mode,
                max_split_per_batch=self.max_split_per_batch,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
            )
            work_metadata = self.work_metadata
            work_indptr = self.work_indptr
            work_info_set = self.work_info_set
            reduce_indptr = self.reduce_indptr
            reduce_final_map = self.reduce_final_map
            reduce_partial_map = self.reduce_partial_map
            num_kv_splits = self.max_split_per_batch

        return self._run_mla_decode(
            q,
            k_buffer,
            layer,
            self.forward_metadata.qo_indptr,
            sparse_kv_indptr,
            sparse_kv_indices,
            sparse_kv_last_page_lens,
            self.forward_metadata.max_q_len,
            work_metadata=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
            flatten_output=True,
        )

    def _forward_decode_mla(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if topk_indices is None:
            topk_indices = self._maybe_build_sparse_topk_fallback(forward_batch)
        if topk_indices is not None:
            return self._forward_decode_mla_sparse(q, layer, k_buffer, topk_indices)

        work_metadata = self.forward_metadata.work_metadata
        work_indptr = self.forward_metadata.work_indptr
        work_info_set = self.forward_metadata.work_info_set
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map
        num_kv_splits = self.forward_metadata.num_kv_splits
        return self._run_mla_decode(
            q,
            k_buffer,
            layer,
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.forward_metadata.kv_last_page_len,
            self.forward_metadata.max_q_len,
            work_metadata=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
            flatten_output=True,
        )
