# TODO: remove all type of argument

# import os

# from atom.model_ops.embed_head import VocabParallelEmbedding
# from atom.model_ops.linear import (
#     QKVParallelLinear,
#     RowParallelLinear,
#     MergedColumnParallelLinear,
#     ColumnParallelLinear,
# )
from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_moe import Qwen3MoeForCausalLM

# global _REGISTERED_ATOM_OPS
# _REGISTERED_ATOM_OPS = {
#     "VocabParallelEmbedding": VocabParallelEmbedding,
#     "QKVParallelLinear": QKVParallelLinear,
#     "RowParallelLinear": RowParallelLinear,
#     "MergedColumnParallelLinear": MergedColumnParallelLinear,
#     "ColumnParallelLinear": ColumnParallelLinear,
# }

_ATOM_SUPPORTED_MODELS = {
    "Qwen3ForCausalLM" : Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM" : Qwen3MoeForCausalLM,
}

# # TODO: remove later
# def _register_custom_op():
#     # register custom op for running the model, which has not been registered by ATOM
#     from vllm.model_executor.custom_op import CustomOp
#     for name, op_cls in _REGISTERED_ATOM_OPS.items():
#         # print('[zejun] ATOM Registering ', op_cls, ' to vllm for ', name, flush=True)
#         CustomOp.register_op(_decorated_op_cls=op_cls, name=name)

def _register_custom_attention():
    from vllm.attention.backends.registry import register_backend, AttentionBackendEnum
    # print('[zejun][PID:', os.getpid(), '] ATOM Registering ATOM custom attention to vllm', flush=True)
    register_backend(backend=AttentionBackendEnum.CUSTOM,
                     is_mamba=False,
                     class_path="atom.model_ops.attentions.aiter_attention.ATOMAttentionBackend")
    # from vllm.attention.selector import global_force_attn_backend
    # global_force_attn_backend(AttentionBackendEnum.CUSTOM)

# TODO: resgiter model for sglang, default for vllm
# def register_custom_model():
    # from vllm import ModelRegistry
    # print('[zejun][atom] Registering ATOM model plugin to vllm', flush=True)

    # # register custom model for ATOM
    # ModelRegistry.register_model(
    #     "Qwen3ForCausalLM",
    #     "atom.models.qwen3:Qwen3ForCausalLM")

    # register custom op
    # _register_custom_op()
    # print('[zejun][atom] Registering ATOM custom op to vllm', flush=True)

    # register custom attention
    # _register_custom_attention()

# TODO: for sgl
# TODO: will fail without init aiter dist, even when tp size is 1
def _init_aiter_dist(config) -> None:
    from aiter import init_dist_env
    from aiter.dist.utils import get_distributed_init_method

    rank = config.parallel_config.rank
    master_ip = config.parallel_config.data_parallel_master_ip
    master_port = config.parallel_config.data_parallel_master_port
    tensor_parallel_size = config.parallel_config.tensor_parallel_size

    distributed_init_method = get_distributed_init_method(master_ip, master_port)

    init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )

# TODO: add for sglang
# TODO: direct register custom op
def _register_ops() -> None:
    '''
    Register custom ops to upper framework
    '''
    _register_custom_attention()
