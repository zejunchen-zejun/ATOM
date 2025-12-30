import torch

from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_moe import Qwen3MoeForCausalLM
from atom.config import Config


_ATOM_SUPPORTED_MODELS = {
    "Qwen3ForCausalLM" : Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM" : Qwen3MoeForCausalLM,
}


def _register_custom_attention_to_vllm() -> None:
    from vllm.attention.backends.registry import register_backend, AttentionBackendEnum
    register_backend(backend=AttentionBackendEnum.CUSTOM,
                     is_mamba=False,
                     class_path="atom.model_ops.attentions.aiter_attention.ATOMAttentionBackend")


def _register_custom_attention_to_sglang() -> None:
    # TODO: register custom attention to sgl
    pass


def _register_ops_to_vllm() -> None:
    '''
    Register custom ops to vllm, including attention
    '''
    _register_custom_attention_to_vllm()


# TODO: register custom attention for sglang
def _register_ops_to_sglang() -> None:
    '''
    Register custom ops to sglang, including attention
    '''
    _register_custom_attention_to_sglang()


def _init_aiter_dist(config: Config) -> None:
    '''
    Initialize aiter dist for using aiter custom collective op
    '''
    from aiter import init_dist_env
    from aiter.dist.utils import get_distributed_init_method

    rank = config.plugin_config.rank
    tensor_parallel_size = config.tensor_parallel_size

    assert config.plugin_config.is_plugin_mode, "Make sure ATOM is running in plugin mode"

    if config.plugin_config.is_vllm:
        dp_master_ip = config.parallel_config.data_parallel_master_ip
        dp_master_port = config.parallel_config.data_parallel_master_port
    elif config.plugin_config.is_sglang:
        if config.plugin_config.sglang_dist_init_addr is not None:
            dp_master_ip, dp_master_port = config.plugin_config.sglang_dist_init_addr.split(":")
        else:
            dp_master_ip = f"127.0.0.1"
            dp_master_port = config.plugin_config.sglang_port_args.nccl_port

    distributed_init_method = get_distributed_init_method(dp_master_ip, dp_master_port)

    init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )
