import os
from typing import Optional
import logging

import atom
from atom.plugin.prepare import _set_framework_backbone

logger = logging.getLogger("atom")

# this flag is used to enable the vllm plugin mode
disable_vllm_plugin = os.getenv("ATOM_DISABLE_VLLM_PLUGIN", "0").lower() == "1"
disable_vllm_plugin_attention = os.getenv("ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", "0").lower() == "1"

# those 2 models are covering most of dense and moe models
ATOM_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMForCausalLM"
ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM"

# when register new model to vllm, add here
# Keys is from hf config arch name
_VLLM_MODEL_REGISTRY_OVERRIDES: dict[str, str] = {
    "Qwen3ForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
}


if not disable_vllm_plugin:
    from vllm.platforms.rocm import RocmPlatform
    logger.info("Enable vLLM plugin mode")
else:
    logger.info("Disable vLLM plugin mode")
    RocmPlatform = object


def _set_plugin_mode() -> None:
    _set_framework_backbone("vllm")


class ATOMPlatform(RocmPlatform):
    # for multi-modality model, for makeing AiterBackend supported by vit
    # get_supported_vit_attn_backends needs to be overridden
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config) -> str:
        # fallback to original behavior of vllm mainline
        if disable_vllm_plugin_attention:
            logger.info("Fallback to original behavior of vLLM mainline")
            return super().get_attn_backend_cls(selected_backend, attn_selector_config)

        # return atom attention backend
        logger.info("Use atom attention backend")
        return "atom.model_ops.attentions.aiter_attention.AiterBackend"


def register_platform() -> Optional[str]:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        return None

    _set_plugin_mode()

    # return the ATOM platform to vllm
    return f"{__name__}.ATOMPlatform"


def patch_model_registry() -> None:
    if disable_vllm_plugin:
        return

    import vllm.model_executor.models.registry as vllm_model_registry

    any_updated = False
    for arch, qual in _VLLM_MODEL_REGISTRY_OVERRIDES.items():
        module_name, class_name = qual.split(":", 1)
        existing = vllm_model_registry.ModelRegistry.models.get(arch)
        if existing is not None:
            # If already overridden to the same target, skip re-registering.
            if (
                getattr(existing, "module_name", None) == module_name
                and getattr(existing, "class_name", None) == class_name
            ):
                continue

        logger.info(f"Register model {arch} to vLLM with {qual}")
        vllm_model_registry.ModelRegistry.register_model(arch, qual)
        any_updated = True

    # clear lru cache
    if any_updated:
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()
