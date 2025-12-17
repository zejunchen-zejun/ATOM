import torch
from vllm.attention.backends.registry import register_backend, AttentionBackendEnum
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl

head_size = 128
dtype = torch.bfloat16
kv_cache_dtype = "auto"
block_size = 16
attn_type = AttentionType.DECODER

class CustomAttentionImpl(AttentionImpl):
    pass

class CustomAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return CustomAttentionImpl

register_backend(backend=AttentionBackendEnum.CUSTOM,
                 is_mamba=False,
                 class_path="CustomAttentionBackend")

attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                block_size,
                use_mla=False,
                has_sink=False,
                attn_type=attn_type,
            )

print('The custom attn_backend is ', attn_backend, flush=True)
