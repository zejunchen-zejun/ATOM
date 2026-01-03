# # SPDX-License-Identifier: MIT
# # Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
from typing import Optional, Any

import torch
from torch import nn

from atom.models.utils import maybe_prefix
from atom.plugin import is_vllm, is_sglang
from atom.config import Config

# Attention class for plugin mode
# It is only a wrapper for constructing and calling the attention
class AttentionForPlugin(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        layer_num=0,
        prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        atom_config = kwargs.get("config", None)
        assert atom_config is not None, "atom_config is required for AttentionForPlugin"

        if is_vllm():
            # use vllm base attention as the custom attention has been 
            # registered to vllm in atom
            from vllm.attention.layer import Attention, AttentionType
            print('[zejun] AttentionForPlugin: using vllm base attention', flush=True)
            assert atom_config is not None, "atom_config is required for plugin mode to vllm"

            # use vllm cache config and quant config to follow the convention of vllm
            cache_config = atom_config.plugin_config.vllm_cache_config
            quant_config = atom_config.plugin_config.vllm_quant_config
            self.attn = Attention(
                num_heads=num_heads,
                head_size=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                alibi_slopes=None,
                prefix=f"{prefix}.attn",
                attn_type=AttentionType.DECODER,
            )
        elif is_sglang():
            # TODO: for now using radix attention as default for sglang
            from sglang.srt.layers.radix_attention import RadixAttention
            print('[zejun] AttentionForPlugin: using sglang radix attention', flush=True)
            self.attn = RadixAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                scaling=scale,
                num_kv_heads=num_kv_heads,
                layer_id=layer_num,
                prefix=maybe_prefix(prefix, "attn"),
            )

    def _forward_vllm(self,
                      query: torch.Tensor,
                      key: torch.Tensor,
                      value: torch.Tensor,
                      **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        # for vllm, model_kwargs is not used because only q k v are passed
        # vllm will dispatch to the attention impl from the backend atom registered
        return self.attn(query, key, value)

    def _forward_sglang(self,
                        query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        # for sglang, forward_batch is required
        forward_batch = model_kwargs.get("forward_batch", None)
        assert forward_batch is not None, "forward_batch is required for sglang"
        return self.attn(q=query,
                         k=key,
                         v=value,
                         forward_batch=forward_batch)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        if is_vllm():
            return self._forward_vllm(query=query,
                                     key=key,
                                     value=value,
                                     **model_kwargs)
        elif is_sglang():
            return self._forward_sglang(query=query,
                                        key=key,
                                        value=value,
                                        **model_kwargs)
