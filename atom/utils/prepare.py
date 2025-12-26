from typing import Any
from functools import lru_cache
from .register import (
    _ATOM_SUPPORTED_MODELS,
    _register_ops_to_vllm,
    _register_ops_to_sglang,
    _init_aiter_dist,
)

_SUPPORTED_FRAMEWORKS = ["vllm", "sglang", "sgl"]
_CURRENT_FRAMEWORK = None

@lru_cache(maxsize=1)
def is_sglang() -> bool:
    global _CURRENT_FRAMEWORK
    if _CURRENT_FRAMEWORK is None:
        raise ValueError("_CURRENT_FRAMEWORK must be set before use")
    return bool(_CURRENT_FRAMEWORK.lower() in ["sglang", "sgl"])

@lru_cache(maxsize=1)
def is_vllm() -> bool:
    global _CURRENT_FRAMEWORK
    if _CURRENT_FRAMEWORK is None:
        raise ValueError("_CURRENT_FRAMEWORK must be set before use")
    return bool(_CURRENT_FRAMEWORK.lower() in ["vllm"])

def _set_framework_backbone(framework: str) -> None:
    if framework.lower() not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework for ATOM: {framework}")
    global _CURRENT_FRAMEWORK
    _CURRENT_FRAMEWORK = framework

# config can be from vllm or sglang
def prepare_model(config: Any, framework: str):
    '''
    Prepare the model to upper framework, including
    register custom ops and init aiter dist
    '''
    print('[zejun] ATOM prepare_model, type config = ', type(config), '. framework = ', framework, flush=True)

    _set_framework_backbone(framework)
    print('[zejun] ATOM prepare_model, is_sglang = ', is_sglang(), flush=True)
    print('[zejun] ATOM prepare_model, is_vllm = ', is_vllm(), flush=True)

    # different framework passed different config
    if is_vllm():
        model_arch = config.model_config.architectures[0]
    elif is_sglang():
        model_arch = config.architectures[0]

    if model_arch not in _ATOM_SUPPORTED_MODELS:
        raise Warning(f"ATOM does not support the required model architecture: {model_arch}")

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    print('[zejun] ATOM prepare_model, model_arch = ', model_arch, ', model_cls = ', model_cls, flush=True)

    if is_vllm():
        _register_ops_to_vllm()
        _init_aiter_dist(config)
    elif is_sglang():
        _register_ops_to_sglang()
        # TODO: init aiter dist for sglang

    # import os
    # pid = os.getpid()
    # from vllm.attention.backends.registry import _ATTN_OVERRIDES
    # for backend, name in _ATTN_OVERRIDES.items():
    #     print('[zejun][', pid, '] ATOM get_model, backend = ', backend, '. name = ', name, flush=True)

    return model_cls(config=config)
