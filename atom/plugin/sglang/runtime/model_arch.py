"""Runtime behavior flags for SGLang plugin model wrappers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArchSpec:
    wrapper_binds_gdn_context: bool = False
    apply_deepseek_patch: bool = False


MODEL_ARCH_SPECS = {
    "DeepseekV3ForCausalLM": ModelArchSpec(apply_deepseek_patch=True),
    "Qwen3MoeForCausalLM": ModelArchSpec(),
    "Qwen3NextForCausalLM": ModelArchSpec(wrapper_binds_gdn_context=True),
}


def get_model_arch_spec(model_arch: str) -> ModelArchSpec:
    return MODEL_ARCH_SPECS.get(model_arch, ModelArchSpec())
