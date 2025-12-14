import os

from atom.model_ops.embed_head import ATOMVocabParallelEmbedding
from atom.model_ops.linear import (
    ATOMQKVParallelLinear,
    ATOMRowParallelLinear,
    ATOMMergedColumnParallelLinear,
    ATOMColumnParallelLinear,
)

global _REGISTERED_ATOM_OPS
_REGISTERED_ATOM_OPS = {
    "VocabParallelEmbedding": ATOMVocabParallelEmbedding,
    "QKVParallelLinear": ATOMQKVParallelLinear,
    "RowParallelLinear": ATOMRowParallelLinear,
    "MergedColumnParallelLinear": ATOMMergedColumnParallelLinear,
    "ColumnParallelLinear": ATOMColumnParallelLinear,
}

def _register_custom_op():
    # register custom op for running the model, which has not been registered by ATOM
    from vllm.model_executor.custom_op import CustomOp
    for name, op_cls in _REGISTERED_ATOM_OPS.items():
        print('[zejun][atom] Registering ', op_cls, ' to vllm for ', name, flush=True)
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

def _register_custom_attention():
    from vllm.attention.backends.registry import register_backend, AttentionBackendEnum
    print('[zejun][PID:', os.getpid(), '][atom] Registering ATOM custom attention to vllm', flush=True)
    register_backend(AttentionBackendEnum.CUSTOM, "atom.model_ops.attentions.aiter_attention.ATOMAttentionBackend")

# TODO: resgiter model for sglang, default for vllm
def register_custom_model():
    from vllm import ModelRegistry
    print('[zejun][atom] Registering ATOM model plugin to vllm', flush=True)

    # register custom model for ATOM
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "atom.models.qwen3:ATOMQwen3ForCausalLM")

    # register custom op
    _register_custom_op()
    print('[zejun][atom] Registering ATOM custom op to vllm', flush=True)

    # register custom attention
    _register_custom_attention()
