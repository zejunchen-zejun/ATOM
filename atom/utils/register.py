
def register_custom_model():
    from vllm import ModelRegistry
    print('[zejun][atom] Registering ATOM model plugin to vllm', flush=True)
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "atom.models.qwen3:ATOMQwen3ForCausalLM")
    print('[zejun][atom] Finish Registering ATOM model plugin to vllm', flush=True)
