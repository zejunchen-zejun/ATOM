from typing import Any
from functools import lru_cache
from .register import (
    _ATOM_SUPPORTED_MODELS,
    _register_ops_to_vllm,
    _register_ops_to_sglang,
    _init_aiter_dist,
)

_SUPPORTED_FRAMEWORKS = ["vllm", "sglang", "sgl"]

@lru_cache(maxsize=1)
def _is_sglang(framework: str) -> bool:
    return framework.lower() in ["sglang", "sgl"]

@lru_cache(maxsize=1)
def _is_vllm(framework: str) -> bool:
    return framework.lower() in ["vllm"]

# config can be from vllm or sglang
def prepare_model(config: Any, framework: str):
    '''
    Prepare the model to upper framework, including
    register custom ops and init aiter dist
    '''
    print('[zejun] ATOM prepare_model, type config = ', type(config), '. framework = ', framework, flush=True)

    if framework.lower() not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework for ATOM: {framework}")

    model_arch = config.model_config.architectures[0]
    if model_arch not in _ATOM_SUPPORTED_MODELS:
        raise Warning(f"ATOM does not support the required model architecture: {model_arch}")

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    print('[zejun] ATOM prepare_model, model_arch = ', model_arch, ', model_cls = ', model_cls, flush=True)

    if _is_vllm(framework):
        _register_ops_to_vllm()
    elif _is_sglang(framework):
        _register_ops_to_sglang()

    if _is_vllm(framework):
        _init_aiter_dist(config)
    elif _is_sglang(framework):
        pass

    # import os
    # pid = os.getpid()
    # from vllm.attention.backends.registry import _ATTN_OVERRIDES
    # for backend, name in _ATTN_OVERRIDES.items():
    #     print('[zejun][', pid, '] ATOM get_model, backend = ', backend, '. name = ', name, flush=True)

    return model_cls(config=config)
