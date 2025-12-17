# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

import torch
from torch import nn

# import torch.distributed as dist
from typing import Any, Optional, Iterable
from transformers import Qwen3Config
from atom.config import Config

from atom.model_ops.activation import SiluAndMul
# from atom.model_ops.attention import Attention
# from atom.model_ops.base_attention import Attention
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ATOMQKVParallelLinear,
    ATOMMergedColumnParallelLinear,
    ATOMRowParallelLinear,
)

# from atom.model_ops.rotary_embedding import get_rope
from aiter.rotary_embedding import get_rope
from atom.model_ops.embed_head import ATOMVocabParallelEmbedding, ParallelLMHead
from atom.config import config_from_vllm
from atom.model_loader.loader import load_model

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
# from vllm.model_executor.models.interfaces import (MixtureOfExperts,
#                                                    SupportsLoRA, SupportsPP)

from vllm.compilation.decorators import support_torch_compile
from vllm.config.vllm import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import maybe_prefix, AutoWeightsLoader
from vllm.attention import Attention, AttentionType
from vllm.config.cache import CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig as VllmQuantizationConfig

class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        kv_cache_dtype: str = "fp16", # TODO: remove because no use
        layer_num: int = 0,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[VllmQuantizationConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().world_size
        self.layer_num = layer_num
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ATOMQKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = ATOMRowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # TODO: concustruct the attention instance
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            alibi_slopes=None,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        print('[zejun] ATOM Qwen3Attention forward', flush=True)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[VllmQuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = ATOMMergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = ATOMRowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        print('[zejun] ATOM Qwen3MLP forward', flush=True)
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        kv_cache_dtype: str = "bf16",
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[VllmQuantizationConfig] = None,
        layer_num: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_num = layer_num

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print('[zejun] ATOM Qwen3DecoderLayer[', self.layer_num, '] forward', flush=True)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
    }
)
class Qwen3Model(nn.Module):

    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = atom_config.model_config.hf_config
        kv_cache_dtype = atom_config.kv_cache_dtype
        self.embed_tokens = ATOMVocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        # TODO: here we use vllm quant config passed from vllm, 
        # but inside the function, the atom specific quant logic will be used
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    kv_cache_dtype=kv_cache_dtype,
                    cache_config=atom_config.cache_config,
                    quant_config=atom_config.vllm_quant_config,
                    layer_num=layer_num,
                    prefix=f"{prefix}.layers.{layer_num}",
                )
                for layer_num in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class ATOMQwen3ForCausalLM(Qwen3ForCausalLM):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(Qwen3ForCausalLM, self).__init__()
        print('[zejun] ATOM ATOMQwen3ForCausalLM init', flush=True)

        # TODO: use original vllm config instead of atom config
        self.atom_config = config_from_vllm(vllm_config)

        self.config = self.atom_config.model_config.hf_config
        self.model = Qwen3Model(
            atom_config=self.atom_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        if self.config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        print('[zejun] ATOM ATOMQwen3ForCausalLM fwd, input_ids = ', input_ids.shape, flush=True)
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: add LogitsProcessor design
        print('[zejun] ATOM call ATOM compute_logits', flush=True)
        logits = self.lm_head(hidden_states)
        print('[zejun] ATOM finish call ATOM compute_logits', flush=True)
        return logits

    # need to provide this method for vllm to load weights for the custom model
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # for name, w in weights:
            # print('[zejun] ATOM load_weights, name = ', name, '. w.shape:', w.shape, '. w.dtype:', w.dtype, flush=True)
        # TODO: weights may consume mem
        loaded_weights_record = load_model(self, self.atom_config)
        # print('[zejun] ATOM loaded_weights_record = \n', loaded_weights_record, flush=True)
        # return to vllm
        return loaded_weights_record
