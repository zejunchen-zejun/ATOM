import warnings

def register_atom_model():
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "atom.models.qwen3:ATOMQwen3ForCausalLM")

try:
    from vllm import ModelRegistry

    # Register models as plugin to vllm
    print('[zejun][atom] Registering ATOM model plugin to vllm', flush=True)
    register_atom_model()
    print('[zejun][atom] Finish Registering ATOM model plugin to vllm', flush=True)

except ImportError:
    warnings.warn("vllm is not installed. Bypass the model registration to vllm", ImportWarning)
