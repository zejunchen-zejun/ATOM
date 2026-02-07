from atom.plugin.prepare import is_sglang

import pdb
import sys
from typing import Any


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
        kv_indptr: torch.Tensor
        kv_indices: torch.Tensor
        qo_indptr: torch.Tensor
        kv_last_page_len: torch.Tensor
        max_q_len: int
        max_kv_len: Optional[int]
        work_metadata: Optional[torch.Tensor] = None
        work_info_set: Optional[torch.Tensor] = None
        work_indptr: Optional[torch.Tensor] = None
        reduce_indptr: Optional[torch.Tensor] = None
        reduce_final_map: Optional[torch.Tensor] = None
        reduce_partial_map: Optional[torch.Tensor] = None
        num_kv_splits: Optional[int] = None
        run_graph: Optional[bool] = True

    class AiterAttnBackendSglplugin(AiterAttnBackend):
        def __init__(
                self,
                model_runner: ModelRunner,
                skip_prefill: bool = False,
                kv_indptr_buf: Optional[torch.Tensor] = None,
            ):
            super().__init__(model_runner, skip_prefill, kv_indptr_buf)


        def init_forward_metadata(self, forward_batch: ForwardBatch):
            """Init auxiliary variables for triton attention backend."""

            bs = forward_batch.batch_size
            kv_indptr = self.kv_indptr
            spec_info = forward_batch.spec_info
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            work_metadata = None
            work_indptr = None
            work_info_set = None
            reduce_indptr = None
            reduce_final_map = None
            reduce_partial_map = None

            num_kv_splits = None
            # num_kv_splits_indptr = None

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

                assert not self.use_mla, "mla is not supported for now."

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                prefix_lens = forward_batch.extend_prefix_lens

                if self.is_multimodal:
                    extend_no_prefix = False
                else:
                    extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                assert not self.use_mla, "mla is not supported for now."

                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
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

            num_kv_splits = None
            # num_kv_splits_indptr = None

            work_metadata = None
            work_info_set = None
            work_indptr = None

            reduce_indptr = None
            reduce_final_map = None
            reduce_partial_map = None

            if forward_mode.is_decode_or_idle():
                qo_indptr = None
                kv_last_page_len = None
                max_q_len = None

                if spec_info is None:
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
                else:
                    kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

                assert not self.use_mla, "mla is not supported for now."

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    kv_indptr[-1].item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    # num_kv_splits_indptr=num_kv_splits_indptr,
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
        ):

            if forward_mode.is_decode_or_idle():
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
            else:
                raise ValueError("Invalid forward mode")

        def prefill_attention(
            self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
        ):

            # variable lenth attention use key value as input
            attn_metadata = fwd_ctx.attn_metadata
            sliding_window = (
                (self.sliding_window, 0, 0)
                if self.sliding_window is not None
                else (-1, -1, 0)
            )
            o = aiter.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=attn_metadata.cu_seqlens_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                max_seqlen_q=attn_metadata.max_seqlen_q,
                max_seqlen_k=attn_metadata.max_seqlen_k,
                min_seqlen_q=attn_metadata.min_seqlen_q,
                dropout_p=attn_metadata.dropout_p,
                softmax_scale=self.scale,
                causal=True,
                window_size=sliding_window,
                sink_ptr=self.sinks,
            )

            return o

        def paged_attention_asm_plugin_mode(
            self,
            q: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            k_scale: torch.Tensor,
            v_scale: torch.Tensor,
            num_decodes: int,
            num_decode_tokens: int,
            attn_metadata: "AttentionMetaData",
            out: torch.Tensor,
        ):
            aiter.pa_fwd_asm(
                Q=q,
                K=k_cache,
                V=v_cache,
                block_tables=attn_metadata.plugin_metadata.block_table[:num_decodes],
                context_lens=attn_metadata.plugin_metadata.seq_lens[:num_decodes],
                block_tables_stride0=attn_metadata.plugin_metadata.block_table[
                    :num_decodes
                ].stride(0),
                K_QScale=k_scale,
                V_QScale=v_scale,
                out_=out[:num_decode_tokens],
                high_precision=0,
            )

            return


        def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            print(f"[DEBUG] call forward_decode from atom!!!")
            return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)

        def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            print(f"[DEBUG] call forward_extend from atom!!!")
            ForkedPdb().set_trace()
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)