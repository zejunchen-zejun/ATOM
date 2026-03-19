"""ATOM DeepSeek model wrapper for SGLang external model loading.

Registers DeepseekV2ForCausalLM and DeepseekV3ForCausalLM as external
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


class DeepseekV2ForCausalLM(nn.Module):
    """ATOM-backed DeepSeek V2/V3 model for SGLang.

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

        # Signal to _init_pools patch: override MLA → MHA with correct dims.
        # Must happen before init_memory_pool runs (which is after model loading).
        from atom.plugin.sglang.oot import _set_mha_override

        head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        _set_mha_override(head_dim, v_head_dim)
        logger.info(
            "ATOM: MHA override registered, head_dim=%d, v_head_dim=%d",
            head_dim,
            v_head_dim,
        )

        import atom

        self.model = atom.prepare_model(config=config, engine="sglang")
        if self.model is None:
            model_arch = getattr(config, "architectures", ["unknown"])[0]
            raise ValueError(
                f"ATOM failed to create model for architecture {model_arch}"
            )

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
        self.model.load_weights(weights)


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]
