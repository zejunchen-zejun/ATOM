# TODO: remove all type of argument

# import os
import torch

# from atom.model_ops.embed_head import VocabParallelEmbedding
# from atom.model_ops.linear import (
#     QKVParallelLinear,
#     RowParallelLinear,
#     MergedColumnParallelLinear,
#     ColumnParallelLinear,
# )
from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_moe import Qwen3MoeForCausalLM
from atom.config import Config

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

def _register_custom_attention_to_vllm() -> None:
    from vllm.attention.backends.registry import register_backend, AttentionBackendEnum
    # print('[zejun][PID:', os.getpid(), '] ATOM Registering ATOM custom attention to vllm', flush=True)
    register_backend(backend=AttentionBackendEnum.CUSTOM,
                     is_mamba=False,
                     class_path="atom.model_ops.attentions.aiter_attention.ATOMAttentionBackend")
    # from vllm.attention.selector import global_force_attn_backend
    # global_force_attn_backend(AttentionBackendEnum.CUSTOM)

def _register_custom_attention_to_sglang() -> None:
    # TODO: register custom attention to sgl
    pass

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

# TODO: for sgl, we cannot get the gpu_id(rank) info from config
# TODO: will fail if aiter dist is not initialized, even when passed tp size is 1
def _init_aiter_dist(config: Config) -> None:
    from aiter import init_dist_env
    from aiter.dist.utils import get_distributed_init_method

    rank = config.parallel_config.rank
    tensor_parallel_size = config.parallel_config.tensor_parallel_size

    if config.is_vllm:
        dp_master_ip = config.parallel_config.data_parallel_master_ip
        dp_master_port = config.parallel_config.data_parallel_master_port
    elif config.is_sglang:
        if config.dist_init_addr is not None:
            dp_master_ip, dp_master_port = config.dist_init_addr.split(":")
        else:
            dp_master_ip = f"127.0.0.1"
            dp_master_port = config.port_args.nccl_port

    distributed_init_method = get_distributed_init_method(dp_master_ip, dp_master_port)

    print('[zejun] ATOM aiter init_dist_env, distributed_init_method = ', distributed_init_method,\
                                          ', tensor_parallel_size = ', tensor_parallel_size,\
                                          ', rank = ', rank,\
                                          ', data_parallel_size=', config.parallel_config.data_parallel_size,\
                                          ', data_parallel_rank=', config.parallel_config.data_parallel_rank,\
                                          flush=True)
    init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )

def _register_ops_to_vllm() -> None:
    '''
    Register custom ops to vllm, including attention
    '''
    _register_custom_attention_to_vllm()

# TODO: add for sglang
def _register_ops_to_sglang() -> None:
    '''
    Register custom ops to sglang, including attention
    '''
    _register_custom_attention_to_sglang()
