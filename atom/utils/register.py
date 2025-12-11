from atom.model_ops.embed_head import ATOMVocabParallelEmbedding

global _REGISTERED_ATOM_OPS
_REGISTERED_ATOM_OPS = {
    "VocabParallelEmbedding": ATOMVocabParallelEmbedding,
}

def _resgiter_custom_op():
    from vllm.model_executor.custom_op import CustomOp
    for name, op_cls in _REGISTERED_ATOM_OPS.items():
        print('[zejun][atom] Registering ', op_cls, ' to vllm for ', name, flush=True)
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

# TODO: resgiter model for sglang, default for vllm
def register_custom_model():
    from vllm import ModelRegistry
    print('[zejun][atom] Registering ATOM model plugin to vllm', flush=True)
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "atom.models.qwen3:ATOMQwen3ForCausalLM")

    _resgiter_custom_op()
    print('[zejun][atom] Registering ATOM custom op to vllm', flush=True)
