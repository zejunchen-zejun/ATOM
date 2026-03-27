# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""

import json
import logging
from typing import Optional, Tuple, Union, Iterable, Any

import torch
from aiter import (
    QuantType,
    cp_gather_indexer_k_quant_cache,
    dtypes,
    gemm_a8w8_blockscale_bpreshuffle,
    get_hip_quant,
    indexer_k_quant_and_cache,
    top_k_per_row_decode,
    top_k_per_row_prefill,
)
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size, get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.fused_fp8_quant import (
    fused_reduce_rms_fp8_group_quant,
    fused_rms_fp8_group_quant,
)
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_reduce_rms_mxfp4_quant,
    fused_rms_mxfp4_quant,
)
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.rotary_embedding import get_rope
from atom.config import (
    CompilationLevel,
    Config,
    QuantizationConfig,
    get_current_atom_config,
)
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.attention_mla import MLAModules, is_rocm_aiter_fp4bmm_enabled
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import LayerNorm, RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
    use_triton_gemm,
)
from atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE, _has_module, quark_post_load_weights
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import (
    is_rocm_aiter_fuse_routed_scaling_factor,
    is_rocm_aiter_fusion_shared_expert_enabled,
)
from atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE, _has_module
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context
from torch import nn
from transformers import PretrainedConfig
from atom.plugin.prepare import is_sglang

# from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.models.deepseek_common.utils import (
    _use_aiter_gfx95,
    _use_aiter,
    _is_gfx95_supported,
    _is_hip,
)
from sglang.srt.layers.quantization.rocm_mxfp4_utils import batched_gemm_afp4wfp4_pre_quant
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)

logger = logging.getLogger("atom")
if use_triton_gemm():
    try:
        from aiter.ops.triton.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle,
        )
        from aiter.ops.triton.gemm_a16w8_blockscale import (
            gemm_a16w8_blockscale_preshuffle,
        )
        from aiter.ops.triton.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
    except ImportError as e:
        logger.warning(
            f"Triton GEMM kernels not available: {e}. Ensure AITER is up-to-date."
        )
        gemm_afp4wfp4_preshuffle = None
        gemm_a16wfp4_preshuffle = None
        gemm_a8w8_blockscale_preshuffle = None
        gemm_a16w8_blockscale_preshuffle = None


from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

from sglang.srt.utils.custom_op import register_custom_op

# TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
# Wrap bmm_fp8 as a custom op so torch.compile does not trace into
# torch.cuda.current_blas_handle() (which returns a non-Tensor).
@register_custom_op(mutates_args=["out"])
def _bmm_fp8_op(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    _bmm_fp8_op(A, B, out, A_scale, B_scale)
    return out

ENABLE_DS_QKNORM_QUANT_FUSION = envs.ATOM_ENABLE_DS_QKNORM_QUANT_FUSION
ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION
ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION = envs.ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION


def _fuse_rmsnorm_fp4_quant_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m, n1 = x1.shape
    n2 = x2.shape[1] if x2 is not None else 0

    out1_quantized = torch.empty((m, n1 // 2), dtype=torch.uint8, device=x1.device)

    scale_n_valid = (n1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE

    scale_m = ((m + 255) // 256) * 256
    scale_n = ((scale_n_valid + 7) // 8) * 8

    out1_bs = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x1.device)

    out2 = None
    if x2 is not None:
        out2 = torch.empty((m, n2), dtype=x1.dtype, device=x1.device)

    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty((m, n1), dtype=x1.dtype, device=x1.device)

    out1_unquantized = None
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


def _fused_rms_fp8_group_quant_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype = dtypes.fp8,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m, n1 = x1.shape
    out1_quantized = torch.empty((m, n1), dtype=dtype_quant, device=x1.device)
    out1_bs = torch.empty(
        (m, (n1 + group_size - 1) // group_size), dtype=torch.float32, device=x1.device
    )
    if transpose_scale:
        out1_bs = out1_bs.transpose(0, 1).contiguous().view(*out1_bs.shape)
    out1_unquantized = None
    if output_unquantized_inp1:
        out1_unquantized = torch.empty_like(x1)
    out2 = None
    if x2 is not None:
        _, n2 = x2.shape
        out2 = torch.empty((m, n2), dtype=x1.dtype, device=x1.device)
    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty((m, n1), dtype=x1.dtype, device=x1.device)
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@torch_compile_guard(gen_fake=_fuse_rmsnorm_fp4_quant_fake)
def _fuse_rmsnorm_fp4_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m = x1.shape[0]

    shuffle_bool = shuffle and (m >= MXFP4_QUANT_BLOCK_SIZE)

    (out1_quantized, out1_bs), _out1_unquantized, out2, out_res1 = (
        fused_rms_mxfp4_quant(
            x1=x1,
            x1_weight=x1_weight,
            x1_epsilon=x1_epsilon,
            x2=x2,
            x2_weight=x2_weight,
            x2_epsilon=0.0 if x2_epsilon is None else x2_epsilon,
            res1=res1,
            shuffle=shuffle_bool,
            scale_shuffle_padding=scale_shuffle_padding,
            output_unquantized_inp1=output_unquantized_inp1,
        )
    )

    out1_unquantized = None
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@torch_compile_guard(gen_fake=_fused_rms_fp8_group_quant_fake)
def _fused_rms_fp8_group_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype = dtypes.fp8,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (out1_quantized, out1_bs), out1_unquantized, out2, out_res1 = (
        fused_rms_fp8_group_quant(
            x1,
            x1_weight,
            x1_epsilon,
            x2,
            x2_weight,
            x2_epsilon,
            group_size,
            dtype_quant,
            res1,
            output_unquantized_inp1,
            transpose_scale,
        )
    )
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


def _fuse_rmsnorm_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype = dtypes.fp8,
    shuffle: bool = True,
    scale_shuffle_padding: bool = False,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
):
    if dtype_quant == dtypes.fp4x2:
        out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = (
            _fuse_rmsnorm_fp4_quant(
                x1,
                x1_weight,
                x1_epsilon,
                x2,
                x2_weight,
                x2_epsilon,
                res1,
                shuffle,
                scale_shuffle_padding,
                output_unquantized_inp1,
            )
        )
    elif dtype_quant == dtypes.fp8:
        out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = (
            _fused_rms_fp8_group_quant(
                x1,
                x1_weight,
                x1_epsilon,
                x2,
                x2_weight,
                x2_epsilon,
                res1,
                dtype_quant,
                group_size,
                output_unquantized_inp1,
                transpose_scale,
            )
        )
    else:
        raise ValueError(
            f"No fused rmsnorm quant kernel availble for quant dtype: {dtype_quant}."
        )
    return (out1_quantized, out1_bs), out1_unquantized, out2, out_res1


def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4_fake(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = True,
    scale_shuffle_padding: Optional[bool] = True,
    output_unquantized_inp1: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]
    device = hidden_states_quant.device
    q_c = torch.empty((M, q_lora_rank // 2), dtype=torch.uint8, device=device)
    scale_n_valid = (q_lora_rank + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    scale_m = ((M + 255) // 256) * 256
    scale_n = ((scale_n_valid + 7) // 8) * 8
    q_c_scale = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=device)
    kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
    k_pe = torch.empty(
        (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )[..., :qk_rope_head_dim]
    return q_c, q_c_scale, kv_c_normed, k_pe


def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8_fake(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    output_unquantized_inp1: Optional[bool] = False,
    transpose_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]
    FP8_QUANT_BLOCK_SIZE = 128
    device = hidden_states_quant.device
    q_c = torch.empty((M, q_lora_rank), dtype=dtypes.fp8, device=device)
    scale_n = (q_lora_rank + FP8_QUANT_BLOCK_SIZE - 1) // FP8_QUANT_BLOCK_SIZE
    q_c_scale = torch.empty((M, scale_n), dtype=dtypes.fp8, device=device)
    kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
    k_pe = torch.empty(
        (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )[..., :qk_rope_head_dim]
    return q_c, q_c_scale, kv_c_normed, k_pe


@torch_compile_guard(
    gen_fake=_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4_fake, mutates_args=[]
)
def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = True,
    scale_shuffle_padding: Optional[bool] = True,
    output_unquantized_inp1: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]

    if hidden_states_quant_scale is None:
        if M <= MXFP4_QUANT_BLOCK_SIZE:
            qkv_lora = gemm_a16wfp4_preshuffle(
                hidden_states_quant,
                weight_qkv_a_proj.view(torch.uint8).view(
                    weight_qkv_a_proj.shape[0] // 16, -1
                ),
                weight_scale_qkv_a_proj.view(torch.uint8).view(
                    weight_scale_qkv_a_proj.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
                ),
                prequant=True,
                skip_reduce=True,
            )
        else:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                hidden_states_quant,
                quant_dtype=dtypes.fp4x2,
                shuffle=(M >= MXFP4_QUANT_BLOCK_SIZE),
            )

            if M >= MXFP4_QUANT_BLOCK_SIZE:
                x_scale = x_scale.view(torch.uint8).view(
                    x_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
                )
            else:
                x_scale = x_scale[:M, ...].view(torch.uint8)

            qkv_lora = gemm_afp4wfp4_preshuffle(
                x.view(torch.uint8),
                weight_qkv_a_proj.view(torch.uint8).view(
                    weight_qkv_a_proj.shape[0] // 16, -1
                ),
                x_scale,
                weight_scale_qkv_a_proj.view(torch.uint8).view(
                    weight_scale_qkv_a_proj.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
                ),
                skip_reduce=True,
            )
    else:
        if M >= MXFP4_QUANT_BLOCK_SIZE:
            hidden_states_quant_scale = hidden_states_quant_scale.view(
                torch.uint8
            ).view(hidden_states_quant_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1)
        else:
            hidden_states_quant_scale = hidden_states_quant_scale[:M, ...].view(
                torch.uint8
            )

        qkv_lora = gemm_afp4wfp4_preshuffle(
            hidden_states_quant.view(torch.uint8),
            weight_qkv_a_proj.view(torch.uint8).view(
                weight_qkv_a_proj.shape[0] // 16, -1
            ),
            hidden_states_quant_scale,
            weight_scale_qkv_a_proj.view(torch.uint8).view(
                weight_scale_qkv_a_proj.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            ),
            skip_reduce=True,
        )

    q_c, kv_c, k_pe = torch.split(
        qkv_lora,
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim],
        dim=-1,
    )

    shuffle_bool = shuffle and (M >= MXFP4_QUANT_BLOCK_SIZE)

    k_pe_reduced = None
    k_pe_reduced_out = None
    if k_pe.dim() == 3:
        device = hidden_states_quant.device
        k_pe_reduced = k_pe
        k_pe_reduced_out = torch.empty(
            (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )[..., :qk_rope_head_dim]
    (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = (
        fused_reduce_rms_mxfp4_quant(
            q_c,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_c,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            k_pe_reduced,
            res1=None,
            shuffle=shuffle_bool,
            scale_shuffle_padding=scale_shuffle_padding,
            output_unquantized_inp1=output_unquantized_inp1,
            dtype=torch.bfloat16,
            out3=k_pe_reduced_out,
        )
    )

    if k_pe_reduced_out is not None:
        k_pe = k_pe_reduced_out
    return q_c, q_c_scale, kv_c_normed, k_pe


@torch_compile_guard(
    gen_fake=_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8_fake, mutates_args=[]
)
def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    output_unquantized_inp1: Optional[bool] = False,
    transpose_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]

    if hidden_states_quant_scale is None:
        if M <= 32:
            qkv_lora = gemm_a16w8_blockscale_preshuffle(
                hidden_states_quant,
                weight_qkv_a_proj.view(weight_qkv_a_proj.shape[0] // 16, -1),
                weight_scale_qkv_a_proj,
                prequant=False,
                skip_reduce=True,
            )
        else:
            quant_func = get_hip_quant(QuantType.per_1x128)
            x, x_scale = quant_func(
                hidden_states_quant,
                quant_dtype=dtypes.fp8,
                transpose_scale=transpose_scale,
            )
            if M <= 128:
                qkv_lora = gemm_a8w8_blockscale_preshuffle(
                    x,
                    weight_qkv_a_proj.view(weight_qkv_a_proj.shape[0] // 16, -1),
                    x_scale,
                    weight_scale_qkv_a_proj,
                    skip_reduce=True,
                )
            else:
                qkv_lora = gemm_a8w8_blockscale_bpreshuffle(
                    x,
                    weight_qkv_a_proj,
                    x_scale,
                    weight_scale_qkv_a_proj,
                    torch.bfloat16,
                )
    else:
        if M <= 128:
            qkv_lora = gemm_a8w8_blockscale_preshuffle(
                hidden_states_quant,
                weight_qkv_a_proj.view(weight_qkv_a_proj.shape[0] // 16, -1),
                hidden_states_quant_scale,
                weight_scale_qkv_a_proj,
                skip_reduce=True,
            )
        else:
            qkv_lora = gemm_a8w8_blockscale_bpreshuffle(
                hidden_states_quant,
                weight_qkv_a_proj,
                hidden_states_quant_scale,
                weight_scale_qkv_a_proj,
                torch.bfloat16,
            )

    q_c, kv_c, k_pe = torch.split(
        qkv_lora,
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim],
        dim=-1,
    )

    k_pe_reduced = None
    k_pe_reduced_out = None
    if k_pe.dim() == 3:
        device = hidden_states_quant.device
        k_pe_reduced = k_pe
        k_pe_reduced_out = torch.empty(
            (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )[..., :qk_rope_head_dim]
    (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = (
        fused_reduce_rms_fp8_group_quant(
            q_c,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_c,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            k_pe_reduced,
            res1=None,
            output_unquantized_inp1=output_unquantized_inp1,
            dtype=torch.bfloat16,
            out3=k_pe_reduced_out,
            transpose_scale=transpose_scale,
        )
    )

    if k_pe_reduced_out is not None:
        k_pe = k_pe_reduced_out

    return q_c, q_c_scale, kv_c_normed, k_pe


def _fuse_qkv_a_proj_reduce_rmsnorm_quant(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype_quant=dtypes.fp8,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = False,
    scale_shuffle_padding: Optional[bool] = False,
    group_size: Optional[int] = 128,
    output_unquantized_inp1: Optional[bool] = False,
    transpose_scale: Optional[bool] = False,
):
    if dtype_quant == dtypes.fp4x2:
        q_c, q_c_scale, kv_c_normed, k_pe = _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4(
            hidden_states_quant,
            weight_qkv_a_proj,
            weight_scale_qkv_a_proj,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
            hidden_states_quant_scale,
            shuffle,
            scale_shuffle_padding,
            output_unquantized_inp1,
        )
    elif dtype_quant == dtypes.fp8:
        q_c, q_c_scale, kv_c_normed, k_pe = _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8(
            hidden_states_quant,
            weight_qkv_a_proj,
            weight_scale_qkv_a_proj,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
            hidden_states_quant_scale,
            output_unquantized_inp1,
            transpose_scale,
        )
    else:
        raise ValueError(
            f"No fused rmsnorm quant kernel availble for quant dtype: {dtype_quant}."
        )

    # logger.info(f"{q_c.shape=}, {q_c_scale.shape=}, {kv_c_normed.shape=}, {k_pe.shape=}, {q_c.stride()=}, {q_c_scale.stride()=}, {kv_c_normed.stride()=}, {k_pe.stride()=}")
    return q_c, q_c_scale, kv_c_normed, k_pe


class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):
    # Using a single shared stream avoids exhausting GPU/HSA resources
    _shared_alt_stream: Optional[torch.cuda.Stream] = None

    @staticmethod
    def _get_shared_stream() -> torch.cuda.Stream:
        if DeepseekV2MoE._shared_alt_stream is None:
            DeepseekV2MoE._shared_alt_stream = torch.cuda.Stream()
        return DeepseekV2MoE._shared_alt_stream

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.reduce_results = reduce_results

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            # MoE gate normally remains unquantized, but may not declare as ignore layers in quantization_config
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts)
            )
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            config=config,
        )

        # Dual-stream support: when mori is enabled,
        # parallelize shared expert and routed expert computation
        self._use_dual_stream = False
        self.alt_stream: Optional[torch.cuda.Stream] = None

        if config.n_shared_experts is not None:
            if (
                not is_rocm_aiter_fusion_shared_expert_enabled()
                and _has_module("mori")
                and get_current_atom_config().compilation_config.level
                != CompilationLevel.PIECEWISE
            ):
                self._use_dual_stream = True
                self.alt_stream = DeepseekV2MoE._get_shared_stream()

            if not is_rocm_aiter_fusion_shared_expert_enabled():
                intermediate_size = (
                    config.moe_intermediate_size * config.n_shared_experts
                )
                self.shared_experts = DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.shared_experts",
                )

    def _forward_dual_stream(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        alt_stream = self.alt_stream

        alt_stream.wait_stream(current_stream)

        # Execute shared experts on current_stream
        shared_output = self.shared_experts(hidden_states)

        # Execute routed experts on alt_stream
        with torch.cuda.stream(alt_stream):
            router_logits = self.gate(hidden_states)
            if hidden_states.dtype != torch.float16:
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, router_logits=router_logits
                )
                if not is_rocm_aiter_fuse_routed_scaling_factor():
                    final_hidden_states = (
                        final_hidden_states * self.routed_scaling_factor
                    )
            else:
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, router_logits=router_logits
                )

        current_stream.wait_stream(alt_stream)

        if hidden_states.dtype != torch.float16:
            final_hidden_states = final_hidden_states + shared_output
        else:
            final_hidden_states = final_hidden_states + shared_output * (
                1.0 / self.routed_scaling_factor
            )

        if self.tp_size > 1 and not ENABLE_ALLREDUCE_RMSNORM_FUSION:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        # Use dual-stream forward when mori is enabled
        DUAL_STREAM_TOKEN_THRESHOLD = 1024
        if (
            self._use_dual_stream
            and self.alt_stream is not None
            and num_tokens > 0
            and num_tokens <= DUAL_STREAM_TOKEN_THRESHOLD
        ):
            return self._forward_dual_stream(hidden_states, num_tokens, hidden_dim)

        if (
            self.n_shared_experts is not None
            and not is_rocm_aiter_fusion_shared_expert_enabled()
        ):
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        if hidden_states.dtype != torch.float16:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
            if not is_rocm_aiter_fuse_routed_scaling_factor():
                final_hidden_states = final_hidden_states * self.routed_scaling_factor
        else:
            # Fix FP16 overflow
            # See DeepseekV2DecoderLayer for more details.
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        if shared_output is not None:
            if hidden_states.dtype != torch.float16:
                final_hidden_states = final_hidden_states + shared_output
            else:
                # Fix FP16 overflow
                # See DeepseekV2DecoderLayer for more details.
                final_hidden_states = final_hidden_states + shared_output * (
                    1.0 / self.routed_scaling_factor
                )
        if self.tp_size > 1 and self.reduce_results:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV32IndexerCache(nn.Module):

    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: str
    ):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    context = forward_context.context
    slot_mapping = attn_metadata.slot_mapping
    # Skip for dummy runs to avoid corrupting KV cache
    if forward_context.context.is_dummy_run:
        # dummy runner
        return weights
    num_decode_tokens = context.batch_size if not context.is_prefill else 0
    indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )
    if context.is_prefill:
        if attn_metadata.max_seqlen_k <= topk_indices_buffer.shape[1]:
            return weights
        prefill_metadata = attn_metadata
        num_prefills = context.batch_size
        total_seq_lens = hidden_states.shape[0]
        k_fp8 = torch.empty(
            [total_seq_lens, head_dim], device=k.device, dtype=dtypes.fp8
        )
        k_scale = torch.empty([total_seq_lens, 1], device=k.device, dtype=torch.float32)
        if prefill_metadata.block_tables.shape[0] < num_prefills:
            new_shape = (num_prefills, prefill_metadata.block_tables.shape[1])
            prefill_metadata.block_tables = torch.full(
                new_shape,
                -1,
                dtype=torch.long,
                device=prefill_metadata.block_tables.device,
            )
        cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),
            prefill_metadata.block_tables,
            prefill_metadata.cu_seqlens_q,
            # num_prefills,
        )
        cu_seqlen_ks = prefill_metadata.cu_seqlen_ks
        cu_seqlen_ke = prefill_metadata.cu_seqlen_ke
        num_tokens = hidden_states.shape[0]
        logits = fp8_mqa_logits(
            Q=q_fp8[num_decode_tokens:num_tokens],
            KV=k_fp8,
            kv_scales=k_scale,
            weights=weights[num_decode_tokens:num_tokens],
            cu_starts=cu_seqlen_ks,
            cu_ends=cu_seqlen_ke,
        )

        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[num_decode_tokens:num_tokens, :topk_tokens]
        top_k_per_row_prefill(
            logits=logits,
            rowStarts=cu_seqlen_ks,
            rowEnds=cu_seqlen_ke,
            indices=topk_indices,
            values=None,
            numRows=num_rows,
            stride0=logits.stride(0),
            stride1=logits.stride(1),
        )
    else:
        decode_metadata = attn_metadata
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
            context.batch_size, -1, *q_fp8.shape[1:]
        )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == context.batch_size
        num_padded_tokens = batch_size * next_n
        batch_size, next_n, heads, _ = padded_q_fp8_decode_tokens.shape
        logits = torch.empty(
            [batch_size * next_n, max_model_len], dtype=torch.float32, device="cuda"
        )
        deepgemm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            logits,
            decode_metadata.context_lens,
            attn_metadata.block_tables,
            max_model_len,
        )
        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.context_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )
    return weights


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    _k_fp8 = _flattened_kv[..., :head_dim].view(torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return weights


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
)


class Indexer(nn.Module):

    def __init__(
        self,
        atom_config: Config,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: Optional[QuantizationConfig],
        cache_config: str,
        topk_indices_buffer: Optional[torch.Tensor],
        prefix: str = "",
    ):
        super().__init__()
        self.atom_config = atom_config
        self.config = config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_func = get_hip_quant(QuantType.per_1x128)
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # TODO (zyongye) change dim to fp8 later to (self.head_dim + 4)
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = atom_config.max_model_len
        self.prefix = prefix
        self.max_total_seq_len = atom_config.max_num_seqs * self.max_model_len
        # register_metadata_builder("indexer_attn_metadata", self.k_cache.get_attn_backend().get_builder_cls())

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: Optional[torch.Tensor],
        positions,
        rotary_emb,
    ) -> torch.Tensor:
        q = self.wq_b(qr, qr_scale)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)

        q_fp8, q_scale = self.quant_func(q, quant_dtype=dtypes.fp8)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = self.weights_proj(hidden_states)
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return torch.ops.aiter.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_num: int = 0,
        topk_indices_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.layer_num = layer_num

        # For FP4 and use_triton_gemm(), fused_qkv_a_proj and q_b_proj are AITER-Triton FP4 GEMMs but o_proj remains AITER BF16 GEMMs,
        # For FP8 and use_triton_gemm(), fused_qkv_a_proj is AITER-Triton FP8 GEMMs while others remain AITER FP8 GEMMs
        q_a_proj_name = (
            "fused_qkv_a_proj" if self.q_lora_rank is not None else "q_a_proj"
        )
        layer_quant_dtype = quant_config.get_layer_quant_config(
            f"{prefix}.{q_a_proj_name}"
        )["quant_dtype"]
        if layer_quant_dtype == dtypes.fp4x2:
            if not use_triton_gemm():
                source_quant_dtype = None
                quant_config = None
                base_quant_config = None
            else:
                source_quant_dtype = torch.bfloat16
                base_quant_config = None
        else:
            source_quant_dtype = None
            base_quant_config = quant_config

        if self.q_lora_rank is not None:
            # self.q_a_proj = ReplicatedLinear(self.hidden_size,
            #                                  self.q_lora_rank,
            #                                  bias=False,
            #                                  quant_config=quant_config,
            #                                  prefix=f"{prefix}.q_a_proj")
            self.fused_qkv_a_proj = MergedReplicatedLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                source_quant_dtype=source_quant_dtype,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
                source_quant_dtype=source_quant_dtype,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
                source_quant_dtype=source_quant_dtype,
            )

            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
                source_quant_dtype=source_quant_dtype,
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=(
                quant_config if is_rocm_aiter_fp4bmm_enabled() else base_quant_config
            ),
            prefix=f"{prefix}.kv_b_proj",
            source_quant_dtype=(
                source_quant_dtype if is_rocm_aiter_fp4bmm_enabled() else None
            ),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=base_quant_config,
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
            prefix=f"{prefix}.o_proj",
            source_quant_dtype=None,
        )

        rope_params = config.rope_parameters
        rope_theta = rope_params.get("rope_theta") or 10000
        # Only use YaRN scaling when config has it (e.g. DeepSeek with factor/type "yarn").
        # GLM-5 has no rope_scaling in config -> use default RoPE (no scaling).
        use_yarn = (
            rope_params.get("factor", 1.0) not in (1.0, None)
            or rope_params.get("type") in ("yarn", "deepseek_yarn")
            or rope_params.get("rope_type") in ("yarn", "deepseek_yarn")
        )
        if use_yarn:
            rope_scaling = dict(rope_params)
            rope_scaling["rope_type"] = "deepseek_yarn"
            if "original_max_position_embeddings" not in rope_scaling:
                factor = float(rope_scaling.get("factor", 1.0))
                rope_scaling["original_max_position_embeddings"] = (
                    int(max_position_embeddings / factor)
                    if factor > 0
                    else max_position_embeddings
                )
        else:
            rope_scaling = None
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.is_v32 = hasattr(config, "index_topk")

        if self.is_v32:
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=True,
            )
            self.indexer = Indexer(
                get_current_atom_config(),
                config,
                hidden_size,
                q_lora_rank,
                base_quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None
        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size

        mla_modules = MLAModules(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            indexer=self.indexer,
        )

        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            use_mla=True,
            mla_modules=mla_modules,
            prefix=prefix,
        )
        self.attn_mha = Attention(
            num_heads=self.num_local_heads,
            head_dim=self.qk_head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_local_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            use_mla=False,
            v_head_dim=self.v_head_dim,
            prefix=maybe_prefix(prefix, "attn_mha"),
        )
        if hasattr(self.attn_mha, "attn"):
            self.attn_mha.attn.kv_b_proj = None

        # When ATOM_ENABLE_DS_QKNORM_QUANT_FUSION is turned on, self.fuse_qknorm_quant is turned on only if FP8 or (use_triton_gemm() and FP4),
        self.prefix = prefix
        self.quant_dtype = None
        self.fuse_qknorm_quant = False
        if quant_config is not None and ENABLE_DS_QKNORM_QUANT_FUSION:
            if layer_quant_dtype == dtypes.fp8 or (
                layer_quant_dtype == dtypes.fp4x2 and use_triton_gemm()
            ):
                self.quant_dtype = layer_quant_dtype
                self.fuse_qknorm_quant = True

        # for sglang
        self.use_nsa = is_deepseek_nsa(config)
        self.use_deep_gemm_bmm = False
        self.alt_stream = None
        self.kv_cache_dtype = cache_config
        self.use_fused_qk_rope_concat_and_cache_mla = _use_aiter_gfx95
        self.current_sgl_plugin_attn_path = None
        # self.w_kc, self.w_vc = self.kv_b_proj.weight.data.unflatten(
        #     0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        # ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.w_kc, self.w_vc = None, None
        self.w_scale = None
        self.w_scale_k = None
        self.w_scale_v = None
        # self.w_kc, self.w_vc = self.kv_b_proj.weight.data.unflatten(
        #     0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        # ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)

    def _forward_sgl_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None
    ) -> torch.Tensor:
        # supplementary code, port from forward_common
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        
        forward_batch = model_kwargs.get("forward_batch", None)
        zero_allocator = model_kwargs.get("zero_allocator", None)
        llama_4_scaling = model_kwargs.get("llama_4_scaling", None)
        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )

            if q.shape[0] != positions.shape[0] and get_tensor_model_parallel_world_size() > 1:
                qkv_lora = torch.cat([q, latent_cache], dim=-1)
                qkv_lora = get_tp_group().all_gather(qkv_lora, dim=0)
                if qkv_lora.shape[0] < positions.shape[0]:
                    raise RuntimeError(
                        f"qkv_lora gather mismatch: got {qkv_lora.shape[0]}, expected {positions.shape[0]}"
                    )
                qkv_lora = qkv_lora[: positions.shape[0]]
                q, latent_cache = torch.split(
                    qkv_lora,
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )

            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                # if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                #     q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                #         q,
                #         self.q_a_layernorm.weight,
                #         self.q_a_layernorm.variance_epsilon,
                #         k_nope,
                #         self.kv_a_layernorm.weight,
                #         self.kv_a_layernorm.variance_epsilon,
                #     )
                # else:
                    q_lora = None
                    _use_aiter_gfx95 = False
                    if (
                        _use_aiter_gfx95
                        and
                        self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        q = self.q_a_layernorm(q)
                        k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_nsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q).view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                topk_indices = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_num,
                )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q).view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_num,
                    )
        else:
            q = self.q_proj(hidden_states).view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        _is_hip= True
        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if (_use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn) or (
                    get_is_capture_mode() and self.w_kc.dtype == torch.float8_e4m3fnuz
                ):
                    # fp8 Triton kernel: always on gfx950,
                    # cudagraph-only on gfx942 (hides launch overhead)
                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        elif self.w_kc.dtype == torch.float8_e4m3fn:
            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                    # if _is_cublas_ge_129
                    # else zero_allocator.allocate(1)
                ),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if self.rotary_emb is not None and not self.use_fused_qk_rope_concat_and_cache_mla:
            assert q_pe.shape[0] == positions.shape[0], (
                f"q_pe tokens {q_pe.shape[0]} != positions {positions.shape[0]}"
            )
            assert k_pe.shape[0] == positions.shape[0], (
                f"k_pe tokens {k_pe.shape[0]} != positions {positions.shape[0]}"
            )
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        # end forward prepare
        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    def _forward_sgl_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,            
    ):
        # 1) build q/k for radix attention path
        save_kv_cache = True

        if self.use_fused_qk_rope_concat_and_cache_mla:
            cos = self.rotary_emb.cos_cache
            sin = self.rotary_emb.sin_cache
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                self.layer_num
            )
            k_scale = self.mla_attn.attn.k_scale

            q, _, k_pe_roped, _ = fused_qk_rope_cat_and_cache_mla(
                q_nope_out,
                q_pe,
                k_nope,
                k_pe,
                kv_cache,
                forward_batch.out_cache_loc,
                positions,
                cos,
                sin,
                k_scale,
                self.rotary_emb.is_neox_style,
                q_out_dtype=q_nope_out.dtype,
            )
            k = torch.cat([k_nope, k_pe_roped], dim=-1)
            save_kv_cache = False
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)

        if llama_4_scaling is not None:
            q = q * llama_4_scaling

        # 2) attention core
        attn_output = self.mla_attn(
            q,
            k,
            k_nope,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        # 3) up-proj by w_vc (port from sglang forward_absorb_core)
        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(attn_output.transpose(0, 1))
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )

        elif _is_hip:
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                y = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    y,
                )
                attn_bmm_output = y.transpose(0, 1).flatten(1, 2)
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    y = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    y = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )
                attn_bmm_output = y.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                torch.zeros((1,), dtype=torch.float32, device=attn_output.device),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        output = self.o_proj(attn_bmm_output)
        return output

    def prepare_qkv_latent(
        self,
        hidden_states: torch.Tensor,
        forward_batch,
    ):
        assert self.q_lora_rank is not None
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        qkv_lora = self.fused_qkv_a_proj(hidden_states, hidden_states_scale)

        # Fallback: when communicator does not enable input_scattered gather,
        # force qkv latent token dimension to align with positions.
        # Use positions.shape[0] (actual input token count) instead of
        # seq_lens_sum (total KV cache length, wrong for decode mode).
        expected_tokens = 0
        if hasattr(forward_batch, "positions") and forward_batch.positions is not None:
            expected_tokens = int(forward_batch.positions.shape[0])
        if expected_tokens <= 0:
            expected_tokens = int(getattr(forward_batch, "seq_lens_sum", 0) or 0)

        if (
            expected_tokens > 0
            and qkv_lora.shape[0] != expected_tokens
            and get_tensor_model_parallel_world_size() > 1
        ):
            qkv_lora = get_tp_group().all_gather(qkv_lora, dim=0)
            if qkv_lora.shape[0] > expected_tokens:
                qkv_lora = qkv_lora[:expected_tokens]
            elif qkv_lora.shape[0] < expected_tokens:
                raise RuntimeError(
                    f"prepare_qkv_latent gather mismatch: got {qkv_lora.shape[0]}, "
                    f"expected {expected_tokens}"
                )
        return qkv_lora


    def _dispatch_sgl_plugin_attn_path(self, forward_batch) -> str:
        if forward_batch.forward_mode.is_extend_without_speculative():
            return "mha"
        return "mla"

    def _forward_sgl_plugin_mode_mla(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None
    ) -> torch.Tensor:
        prepared = self._forward_sgl_prepare(positions, hidden_states, **model_kwargs)
        return self._forward_sgl_core(*prepared)

    def _get_sglang_radix_attn(self, attn_module):
        return attn_module.attn if hasattr(attn_module, "attn") else attn_module

    def _set_mla_kv_buffer_for_mha(
        self,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch,
    ) -> None:
        attn_mha = self._get_sglang_radix_attn(self.attn_mha)
        cache_k = torch.cat([kv_a.unsqueeze(1), k_pe], dim=-1)
        # For staged migration, keep the legacy MLATokenToKVPool write contract:
        # write a single concatenated latent cache tensor via set_kv_buffer.
        # MLATokenToKVPool ignores cache_v in this path.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            attn_mha,
            forward_batch.out_cache_loc,
            cache_k,
            cache_k,
        )

    def _can_run_sgl_mha_now(self, forward_batch) -> bool:
        # For aiter backend, public SGLang keeps prefill on the MHA path even
        # when prefix cache is present. The backend consumes MLA latent cache
        # metadata and reconstructs the needed K/V from KV cache as needed.
        if self.use_nsa:
            return False
        if self.kv_b_proj.weight.dtype == torch.uint8:
            return False
        return True

    def _forward_sgl_mha_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ):
        forward_batch = model_kwargs.get("forward_batch", None)
        if forward_batch is None:
            raise RuntimeError("forward_batch is required in _forward_sgl_mha_prepare")

        attn_mha = self._get_sglang_radix_attn(self.attn_mha)
        if getattr(attn_mha, "kv_b_proj", None) is None:
            attn_mha.kv_b_proj = self.kv_b_proj

        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )

            if (
                q.shape[0] != positions.shape[0]
                and get_tensor_model_parallel_world_size() > 1
            ):
                qkv_lora = torch.cat([q, latent_cache], dim=-1)
                qkv_lora = get_tp_group().all_gather(qkv_lora, dim=0)
                if qkv_lora.shape[0] < positions.shape[0]:
                    raise RuntimeError(
                        f"qkv_lora gather mismatch: got {qkv_lora.shape[0]}, expected {positions.shape[0]}"
                    )
                qkv_lora = qkv_lora[: positions.shape[0]]
                q, latent_cache = torch.split(
                    qkv_lora,
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )

            if (
                _use_aiter_gfx95
                and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
            ):
                (q, q_scale), _, _, _ = fused_rms_fp8_group_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.eps,
                    None,
                    None,
                    None,
                    group_size=128,
                    dtype_quant=torch.float8_e4m3fn,
                    res1=None,
                    output_unquantized_inp1=False,
                )
                q = self.q_b_proj(q, q_scale).view(-1, self.num_local_heads, self.qk_head_dim)
            else:
                q = self.q_a_layernorm(q)
                q = self.q_b_proj(q).view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states).view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)

        if _use_aiter_gfx95 and self.kv_b_proj.weight.dtype == torch.float8_e4m3fn:
            (kv_a_quanted, kv_a_quanted_scale), kv_a, _, _ = fused_rms_fp8_group_quant(
                kv_a,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.eps,
                None,
                None,
                None,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                res1=None,
                output_unquantized_inp1=True,
            )
        else:
            kv_a_quanted = None
            kv_a = self.kv_a_layernorm(kv_a)

        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        self._set_mla_kv_buffer_for_mha(kv_a, k_pe, forward_batch)

        if kv_a_quanted is not None:
            kv = self.kv_b_proj(kv_a_quanted, kv_a_quanted_scale)
        else:
            kv = self.kv_b_proj(kv_a)
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k = torch.cat(
            [k_nope, k_pe.expand(-1, self.num_local_heads, -1)],
            dim=-1,
        )
        return q, k, v, forward_batch

    def _forward_sgl_mha_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor:
        attn_output = self.attn_mha(
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=False,
        )
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        return self.o_proj(attn_output)

    def _forward_sgl_plugin_mode_mha(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None
    ) -> torch.Tensor:
        forward_batch = model_kwargs.get("forward_batch", None)
        if forward_batch is None:
            raise RuntimeError("forward_batch is required in _forward_sgl_plugin_mode_mha")
        if not self._can_run_sgl_mha_now(forward_batch):
            self.current_sgl_plugin_attn_path = "mla_fallback"
            return self._forward_sgl_plugin_mode_mla(
                positions,
                hidden_states,
                **model_kwargs,
            )
        prepared = self._forward_sgl_mha_prepare(
            positions,
            hidden_states,
            **model_kwargs,
        )
        return self._forward_sgl_mha_core(*prepared)

    def forward_sgl_plugin_mode(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None
    ) -> torch.Tensor:
        forward_batch = model_kwargs.get("forward_batch", None)
        if forward_batch is None:
            raise RuntimeError("forward_batch is required in forward_sgl_plugin_mode")

        attn_tp_context = get_attn_tp_context()
        with attn_tp_context.maybe_input_scattered(forward_batch):
            if self.q_lora_rank is not None:
                attn_tp_context.set_attn_inputs(
                    AttentionInputs(
                        hidden_states,
                        forward_batch,
                        self.prepare_qkv_latent,
                    )
                )

            attn_path = self._dispatch_sgl_plugin_attn_path(forward_batch)
            self.current_sgl_plugin_attn_path = attn_path
            if attn_path == "mha":
                return self._forward_sgl_plugin_mode_mha(
                    positions,
                    hidden_states,
                    **model_kwargs,
                )
            if attn_path == "mla":
                return self._forward_sgl_plugin_mode_mla(
                    positions,
                    hidden_states,
                    **model_kwargs,
                )
            raise ValueError(f"Unsupported plugin attention path: {attn_path}")

    def forward_common(

        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None
    ):
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states

        if self.q_lora_rank is not None:
            if self.fuse_qknorm_quant and use_triton_gemm():
                q_c, q_c_scale, kv_c_normed, k_pe = (
                    _fuse_qkv_a_proj_reduce_rmsnorm_quant(
                        hidden_states,
                        self.fused_qkv_a_proj.weight,
                        self.fused_qkv_a_proj.weight_scale,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.eps,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.eps,
                        self.q_lora_rank,
                        self.kv_lora_rank,
                        self.qk_rope_head_dim,
                        dtype_quant=self.quant_dtype,
                        hidden_states_quant_scale=hidden_states_scale,
                        shuffle=True,
                        scale_shuffle_padding=True,
                        group_size=128,
                        output_unquantized_inp1=False,
                        transpose_scale=True,
                    )
                )
                hidden_states_or_q_c = q_c
                hidden_states_or_q_c_scale = q_c_scale
            else:
                qkv_lora = self.fused_qkv_a_proj(hidden_states, hidden_states_scale)
                # ckq = self.q_a_proj(hidden_states)
                q_c, kv_c, k_pe = torch.split(
                    qkv_lora,
                    [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
                    dim=-1,
                )
                # fuse q_c norm + kv_c norm + quant of hidden_states_or_q_c
                if self.fuse_qknorm_quant:
                    (
                        (hidden_states_or_q_c, hidden_states_or_q_c_scale),
                        _,
                        kv_c_normed,
                        _,
                    ) = _fuse_rmsnorm_quant(
                        q_c,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.eps,
                        kv_c,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.eps,
                        None,
                        dtype_quant=self.quant_dtype,
                        shuffle=False,
                        scale_shuffle_padding=False,
                        group_size=128,
                        output_unquantized_inp1=False,
                        transpose_scale=True,
                    )
                else:
                    hidden_states_or_q_c = self.q_a_layernorm(q_c)
        else:
            hidden_states_or_q_c = hidden_states
            kv_c, k_pe = torch.split(
                self.kv_a_proj_with_mqa(hidden_states, hidden_states_scale),
                [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1,
            )
        if not self.fuse_qknorm_quant:
            kv_c_normed = self.kv_a_layernorm(kv_c)
            hidden_states_or_q_c_scale = None
        if self.is_v32 and self.indexer is not None:
            _topk_indices = self.indexer(
                hidden_states,
                hidden_states_or_q_c,
                hidden_states_or_q_c_scale,
                positions,
                self.indexer_rope_emb,
            )

        return self.mla_attn(
            hidden_states_or_q_c,
            kv_c_normed,
            k_pe,
            positions,
            hidden_states_or_q_c_scale,
        )       

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None
    ) -> torch.Tensor:
        if is_sglang():
            attn_output = self.forward_sgl_plugin_mode(positions, hidden_states, **model_kwargs)
        else:
            attn_output = self.forward_common(positions, hidden_states, **model_kwargs)
        return attn_output

    def process_weights_after_loading(self) -> None:
        # only for sglang plugin mode
        if not is_sglang():
            return
        self._process_mla_kv_b_proj_after_loading_sgl()

    def _process_mla_kv_b_proj_after_loading_sgl(self) -> None:
        # lazy imports: only needed for sglang plugin path
        from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz
        from sglang.srt.layers.quantization.fp8_utils import (
            block_quant_dequant,
            block_quant_to_tensor_quant,
            channel_quant_to_tensor_quant,
            inverse_transform_scale_ue8m0,
        )
        from sglang.srt.layers.quantization.int8_utils import (
            block_dequant as int8_block_dequant,
        )
        from sglang.srt.layers.deep_gemm_wrapper import (
            ENABLE_JIT_DEEPGEMM,
            DEEPGEMM_BLACKWELL,
        )
        from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
        from sglang.srt.models.deepseek_common.utils import (
            _is_cpu,
            _is_cpu_amx_available,
            _is_cuda,
            _is_fp8_fnuz,
            _is_hip,
            _is_npu,
            _use_aiter_gfx95,
            awq_dequantize_func,
        )
        from sglang.srt.utils import bind_or_assign, get_bool_env_var

        # read kv_b_proj weight (awq compatible)
        if hasattr(self.kv_b_proj, "qweight"):
            awq_dequantize_f = awq_dequantize_func()
            if awq_dequantize_f is None:
                raise ValueError("AWQ dequantize function is not supported for current device")
            w = awq_dequantize_f(
                self.kv_b_proj.qweight,
                self.kv_b_proj.scales,
                self.kv_b_proj.qzeros,
            ).T
        else:
            w = self.kv_b_proj.weight

        # On ROCm, ATOM creates parameters with fnuz dtype but loads fn bytes
        # into them (weight_loader_process view-casts a detached copy, leaving
        # the nn.Parameter as fnuz). At this point LinearBase's
        # process_weights_after_loading hasn't run yet (parent module iterates
        # before child in named_modules). View-cast back to fn so the
        # normalize path works correctly.
        if _is_fp8_fnuz and w.dtype == torch.float8_e4m3fnuz:
            w = w.view(torch.float8_e4m3fn)

        use_deep_gemm_bmm = False
        block_scale = None
        weight_block_size = None

        # Derive weight_block_size from ATOM's quant_type system
        from aiter import QuantType as _AiterQuantType
        _atom_qt = getattr(self.kv_b_proj, "quant_type", None)
        if _atom_qt == _AiterQuantType.per_1x128:
            weight_block_size = [128, 128]
        elif _atom_qt == _AiterQuantType.per_1x32:
            weight_block_size = [1, 32]

        # fp8 path
        if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            if weight_block_size is not None:
                assert hasattr(self.kv_b_proj, "weight_scale_inv") or hasattr(self.kv_b_proj, "weight_scale")
                weight_scale = (
                    self.kv_b_proj.weight_scale
                    if hasattr(self.kv_b_proj, "weight_scale")
                    else self.kv_b_proj.weight_scale_inv
                )

                if _is_fp8_fnuz and w.dtype == torch.float8_e4m3fn:
                    weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                        weight=w,
                        weight_scale=weight_scale,
                        input_scale=None,
                    )
                else:
                    weight = w

                if (
                    should_deepgemm_weight_requant_ue8m0(
                        weight_block_size=weight_block_size
                    )
                    and getattr(weight_scale, "format_ue8m0", False)
                ):
                    weight_scale = inverse_transform_scale_ue8m0(weight_scale, mn=weight.shape[-2])

                if _is_cuda and weight_block_size[0] == 128 and weight_block_size[1] == 128:
                    if (
                        ENABLE_JIT_DEEPGEMM
                        and not DEEPGEMM_BLACKWELL
                        and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                    ):
                        block_scale = weight_scale
                        use_deep_gemm_bmm = True
                    else:
                        w = block_quant_dequant(
                            weight,
                            weight_scale,
                            weight_block_size,
                            torch.bfloat16,
                        )
                else:
                    w, scale = block_quant_to_tensor_quant(weight, weight_scale, weight_block_size)
                    self.w_scale = scale
            else:
                if w.dtype == torch.float8_e4m3fn and _is_fp8_fnuz:
                    weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                        weight=w,
                        weight_scale=self.kv_b_proj.weight_scale,
                        input_scale=None,
                    )
                else:
                    weight = w
                    weight_scale = self.kv_b_proj.weight_scale

                w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                self.w_scale = scale

        # int8 path
        if w.dtype == torch.int8:
            if weight_block_size is not None:
                assert hasattr(self.kv_b_proj, "weight_scale_inv")
                w = int8_block_dequant(
                    w,
                    self.kv_b_proj.weight_scale_inv,
                    weight_block_size,
                ).to(torch.bfloat16)
            else:
                w = w.to(torch.bfloat16) * self.kv_b_proj.weight_scale.to(torch.bfloat16)

        # split to kc/vc
        w_kc, w_vc = w.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)

        # quark fp4 special path (ATOM utility)
        quant_method = getattr(self.kv_b_proj, "quant_method", None)
        quant_config = getattr(quant_method, "quant_config", None)
        if _use_aiter_gfx95 and quant_config is not None and quant_config.get_name() == "quark":
            w_kc, self.w_scale_k, w_vc, self.w_scale_v = quark_post_load_weights(self, w, "mxfp4")

        if not use_deep_gemm_bmm:
            self.w_kc = bind_or_assign(
                self.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            )
            w_vc = w_vc.contiguous().transpose(1, 2)
            if _is_npu:
                w_vc = w_vc.contiguous()
            self.w_vc = bind_or_assign(self.w_vc, w_vc)

            if hasattr(self.kv_b_proj, "weight_scale") and self.w_scale is None:
                self.w_scale = bind_or_assign(self.w_scale, self.kv_b_proj.weight_scale)
                if _is_hip:
                    self.w_scale *= 2.0

            if _is_cpu and _is_cpu_amx_available and w.dtype == torch.float8_e4m3fn:
                self.w_kc = self.w_kc.to(torch.bfloat16) * self.w_scale
                self.w_vc = self.w_vc.to(torch.bfloat16) * self.w_scale
        else:
            num_tiles_k = self.qk_nope_head_dim // weight_block_size[1]
            num_tiles_n = self.v_head_dim // weight_block_size[0]
            ws_kc, ws_vc = block_scale.unflatten(0, (-1, (num_tiles_k + num_tiles_n))).split(
                [num_tiles_k, num_tiles_n], dim=1
            )

            self.w_scale_k = bind_or_assign(self.w_scale_k, ws_kc.transpose(1, 2).contiguous())
            self.w_scale_v = bind_or_assign(self.w_scale_v, ws_vc.contiguous())
            self.w_kc = bind_or_assign(self.w_kc, w_kc.transpose(1, 2).contiguous())
            self.w_vc = bind_or_assign(self.w_vc, w_vc.contiguous())
            self.use_deep_gemm_bmm = True


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        topk_indices_buffer: Optional[torch.Tensor] = None,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        layer_num: int = 0,
        is_mtp_block: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = DeepseekV2MLAAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
            topk_indices_buffer=topk_indices_buffer,
        )

        # When ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is turned on self.fuse_input_norm_quant is turned on only if use_triton_gemm and (FP8 or FP4),
        # Because AR_RMS and RMS_Quant cannot co-exist for input_layernorm, this block of codes ensures 3 things when ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is turned on:
        #   1. RMS_Quant fusion is only used for input_layernorm
        #   2. The reduce_results variable is re-enabled for feed forward layers (MOE and MLP), because AR_RMS is now disabled in the beginning of the next layer
        #   3. AR_RMS is turned off for input_layernorm but still enabled for post_attention_layernorm if ENABLE_ALLREDUCE_RMSNORM_FUSION is turned on
        self.quant_dtype = (
            None
            if quant_config is None
            else quant_config.global_quant_config["quant_dtype"]
        )
        self.fuse_input_norm_quant = False
        self.fuse_ar_input_norm = ENABLE_ALLREDUCE_RMSNORM_FUSION
        if quant_config is not None and ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION:
            if (
                self.quant_dtype == dtypes.fp8 or self.quant_dtype == dtypes.fp4x2
            ) and use_triton_gemm():
                self.fuse_input_norm_quant = True
                if self.fuse_ar_input_norm:
                    self.fuse_ar_input_norm = False
                    if layer_idx == 0:
                        logger.info(
                            "Warning: Because ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is turned on, AR + RMS fusion is turned off for input_layernorm and reduce_results is re-enabled for first k dense layer down_proj"
                        )
            else:
                if layer_idx == 0:
                    logger.info(
                        "Info: Because ATOM_USE_TRITON_GEMM is not turned on in DeepSeek-R1, ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is turned off automatically"
                    )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                reduce_results=not self.fuse_ar_input_norm,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not self.fuse_ar_input_norm,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=self.fuse_ar_input_norm
            and self.layer_idx > 0
            and not is_mtp_block,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )
        self.routed_scaling_factor = config.routed_scaling_factor
        self.fuse_rmsnorm_quant = (
            ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION and self.quant_dtype is not None
        )
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **model_kwargs: dict[str, Any] | None
    ) -> torch.Tensor:
        # Self Attention
        if self.fuse_input_norm_quant:
            assert self.quant_dtype is not None
            weight = self.input_layernorm.weight
            eps = self.input_layernorm.eps
            if residual is None:
                residual = hidden_states
                (hidden_states_quant, hidden_states_quant_scale), _, _, _ = (
                    _fuse_rmsnorm_quant(
                        hidden_states,
                        weight,
                        eps,
                        None,
                        None,
                        None,
                        None,
                        dtype_quant=self.quant_dtype,
                        shuffle=True,
                        scale_shuffle_padding=True,
                        group_size=128,
                        output_unquantized_inp1=False,
                        transpose_scale=True,
                    )
                )
            else:
                (hidden_states_quant, hidden_states_quant_scale), _, _, residual = (
                    _fuse_rmsnorm_quant(
                        hidden_states,
                        weight,
                        eps,
                        None,
                        None,
                        None,
                        residual,
                        dtype_quant=self.quant_dtype,
                        shuffle=True,
                        scale_shuffle_padding=True,
                        group_size=128,
                        output_unquantized_inp1=False,
                        transpose_scale=True,
                    )
                )

            hidden_states = (hidden_states_quant, hidden_states_quant_scale)

        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **model_kwargs,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


@support_torch_compile
class DeepseekV2Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = DeepseekV2DecoderLayer,
    ):
        super().__init__()

        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                atom_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: DeepseekV2DecoderLayer(
                config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        # fused_allreduce will have to be turned off here if the fuse_ar_input_norm variable is False in the last layer
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                fused_allreduce=self.layers[self.end_layer - 1].fuse_ar_input_norm,
            )
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **model_kwargs: dict[str, Any] | None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual, **model_kwargs)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if is_rocm_aiter_fusion_shared_expert_enabled()
                else 0
            ),
        )


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = DeepseekV2DecoderLayer,
    ):
        super().__init__()
        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.atom_config = atom_config

        if hasattr(config, "q_lora_rank") and config.q_lora_rank is not None:
            self.packed_modules_mapping = {
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }
        else:
            self.packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }

        self.model = DeepseekV2Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        if is_sglang():
            from sglang.srt.configs.model_config import is_deepseek_nsa
            get_attn_tp_context().init_context(config.q_lora_rank, is_deepseek_nsa(config))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **model_kwargs: dict[str, Any] | None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds,
            **model_kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # load weights in plugin mode and discard passed weights generator
        # here prefix is "model." because Qwen3MoeForCausalLM is constructed in model
        # wrapper class, so the name of loaded weights are prefixed with "model.".
        # The vLLM will check the name of the loaded weights to make sure all the
        # weights are loaded correctly
        
        # lazy import to avoid circular import issue since model_loader also imports model..
        from atom.model_loader.loader import load_model_in_plugin_mode
        loaded_weights_record = load_model_in_plugin_mode(
            model=self, config=self.atom_config, prefix="model."
        )
        return loaded_weights_record

class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    """GLM 5.0 MoE (structurally similar to DeepSeek v3.2). Reuses DeepseekV2 implementation."""

    pass
