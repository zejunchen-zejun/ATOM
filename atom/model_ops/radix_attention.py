# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import nn
from typing import Optional

from .attention_mla import MLAModules
from .base_attention import BaseAttention
from atom.plugin.prepare import is_plugin_mode, is_sglang
from atom.models.utils import maybe_prefix
from atom.utils import envs


class RadixAttention(BaseAttention):
    """Attention wrapper for sglang plugin mode.

    Delegates to sglang's RadixAttention internally, adapting ATOM's
    attention interface to sglang's forward_batch-based API.
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        per_layer_sliding_window: Optional[int] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=use_mla,
            mla_modules=mla_modules,
            sinks=sinks,
            per_layer_sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
            prefix=prefix,
            **kwargs,
        )

        if is_sglang():
            from sglang.srt.layers.radix_attention import RadixAttention

            explicit_v_head_dim = kwargs.get("v_head_dim", None)
            if explicit_v_head_dim is not None:
                _v_head_dim = explicit_v_head_dim
            elif use_mla and mla_modules is not None:
                _v_head_dim = mla_modules.kv_lora_rank
            else:
                _v_head_dim = head_dim

            self.attn = RadixAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                scaling=scale,
                num_kv_heads=num_kv_heads,
                layer_id=layer_num,
                v_head_dim=_v_head_dim,
                prefix=maybe_prefix(prefix, "attn"),
            )
            # sglang's RadixAttention expects k_scale/v_scale on device;
            # ensure they exist with identity scaling for non-quantised KV cache.
            if self.attn.k_scale is None:
                self.attn.k_scale = torch.nn.Parameter(
                    torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    requires_grad=False,
                )
            if self.attn.v_scale is None:
                self.attn.v_scale = torch.nn.Parameter(
                    torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    requires_grad=False,
                )
        else:
            raise NotImplementedError(
                "RadixAttention is only supported for plugin mode for sglang for now"
            )
        # if True, save cache will be done in rope
        self.use_aiter_rope_fused_qknorm = envs.ATOM_ROPE_FUSED_QKNORM

    def forward_impl_plugin_mode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata=None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        positions: torch.Tensor = None,
        q_scale: torch.Tensor = None,
        **kwargs,
    ):
        if is_sglang():
            # for sglang, forward_batch is required
            forward_batch = kwargs.get("forward_batch", None)
            # When fused rope+qknorm is active, KV cache is saved inside the
            # fused kernel, so we skip the separate save step in sglang's attn.
            save_kv_cache = kwargs.get("save_kv_cache", not self.use_aiter_rope_fused_qknorm)
            assert forward_batch is not None, "forward_batch is required for sglang"
            # forward_batch contains the filed attn_backend, which will find the backend registered in ATOM
            return self.attn(
                query,
                key,
                value,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
            )
        else:
            raise NotImplementedError(
                "RadixAttention is only supported for plugin mode for sglang for now"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if is_plugin_mode():
            o = self.forward_impl_plugin_mode(
                query=query,
                key=key,
                value=value,
                positions=positions,
                q_scale=q_scale,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "RadixAttention is not supported for server mode for now"
            )
        return o
