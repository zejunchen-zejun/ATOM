import warnings

try:
    from vllm import ModelRegistry

    # Register models as plugin to vllm
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "atom.models.qwen3:Qwen3ForCausalLM")

except ImportError:
    warnings.warn("vllm is not installed. Bypass the model registration to vllm", ImportWarning)
