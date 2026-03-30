"""ATOM GLM MoE wrappers for SGLang external model loading.

Registers Glm4MoeForCausalLM and GlmMoeDsaForCausalLM as external
model classes via SGLANG_EXTERNAL_MODEL_PACKAGE, replacing sglang's
built-in implementations with ATOM-optimized versions.
"""

import logging
from typing import Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger("atom.plugin.sglang.oot")


def _patch_rope_in_attention_layers(atom_model):
    """Inject RoPE into each Glm4MoeAttention's inner RadixAttention.

    The ATOM GLM4 model has RoPE commented out in Glm4MoeAttention.forward().
    This function wraps each ATOM RadixAttention.forward() to apply rotary_emb
    before delegating, without modifying any ATOM source files.
    """
    from atom.models.glm4_moe import Glm4MoeAttention

    patched = 0
    for module in atom_model.modules():
        if not isinstance(module, Glm4MoeAttention):
            continue

        inner_attn = module.attn
        original_forward = inner_attn.forward
        rotary_emb = module.rotary_emb

        def _make_rope_wrapper(orig_fwd, rope):
            def forward_with_rope(query, key, value, positions=None, **kwargs):
                if positions is not None:
                    query, key = rope(positions, query, key)
                return orig_fwd(query, key, value, positions, **kwargs)

            return forward_with_rope

        inner_attn.forward = _make_rope_wrapper(original_forward, rotary_emb)
        patched += 1

    logger.info("Patched RoPE into %d Glm4MoeAttention layers", patched)


def _needs_glm4_rope_patch(config) -> bool:
    architectures = getattr(config, "architectures", None) or []
    return bool(architectures and architectures[0] == "Glm4MoeForCausalLM")


class Glm4MoeForCausalLM(nn.Module):
    """ATOM-backed GLM MoE model wrapper for SGLang.

    This wrapper delegates model creation and weight loading to ATOM's
    plugin system, while conforming to sglang's model interface
    (forward signature, LogitsProcessorOutput return type, load_weights).
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Initializing ATOM backend for %s", self.__class__.__name__)

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        import atom

        self.model = atom.prepare_model(config=config, engine="sglang")
        if self.model is None:
            model_arch = getattr(config, "architectures", ["unknown"])[0]
            raise ValueError(
                f"ATOM failed to create model for architecture {model_arch}"
            )

        if _needs_glm4_rope_patch(config):
            _patch_rope_in_attention_layers(self.model)

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=input_embeds,
            forward_batch=forward_batch,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.model.lm_head,
                forward_batch,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        from atom.model_loader.loader import load_model_in_plugin_mode

        return load_model_in_plugin_mode(
            model=self.model, config=self.model.atom_config, prefix="model."
        )


class GlmMoeDsaForCausalLM(Glm4MoeForCausalLM):
    """ATOM-backed GLM-5 wrapper for SGLang.

    GLM-5 reuses ATOM's DeepSeekV2 implementation, so it only needs the
    generic SGLang wrapper and must skip the GLM4-specific RoPE patch.
    """


EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
