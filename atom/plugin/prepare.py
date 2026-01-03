from typing import Any

_SUPPORTED_FRAMEWORKS = ["vllm", "sglang", "sgl"]
_CURRENT_FRAMEWORK = None


def is_sglang() -> bool:
    global _CURRENT_FRAMEWORK
    if _CURRENT_FRAMEWORK is None:
        raise ValueError("_CURRENT_FRAMEWORK must be set before use")
    return bool(_CURRENT_FRAMEWORK.lower() in ["sglang", "sgl"])


def is_vllm() -> bool:
    global _CURRENT_FRAMEWORK
    if _CURRENT_FRAMEWORK is None:
        raise ValueError("_CURRENT_FRAMEWORK must be set before use")
    return bool(_CURRENT_FRAMEWORK.lower() in ["vllm"])


def is_plugin_mode() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK is not None)


def _set_framework_backbone(framework: str) -> None:
    if framework.lower() not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework {framework} for ATOM to plug in")
    global _CURRENT_FRAMEWORK
    _CURRENT_FRAMEWORK = framework


# TODO: remove the framework argument, get this info in sys argv
def prepare_model(config: Any, framework: str):
    '''
    Prepare the model to upper framework, including
    register custom ops and init aiter dist
    '''
    print('[zejun] ATOM prepare_model, type config = ', type(config), '. framework = ', framework, flush=True)

    _set_framework_backbone(framework)
    print('[zejun] ATOM prepare_model, is_plugin_mode = ', is_plugin_mode(), flush=True)
    print('[zejun] ATOM prepare_model, is_sglang = ', is_sglang(), flush=True)
    print('[zejun] ATOM prepare_model, is_vllm = ', is_vllm(), flush=True)

    # different framework passed different config
    if is_vllm():
        model_arch = config.model_config.architectures[0]
    elif is_sglang():
        model_arch = config.architectures[0]

    # import here to avoid partial initialization
    from .register import (
        _ATOM_SUPPORTED_MODELS,
        _register_ops_to_vllm,
        _register_ops_to_sglang,
        _init_aiter_dist,
    )

    if model_arch not in _ATOM_SUPPORTED_MODELS:
        raise Warning(f"ATOM does not support the required model architecture: {model_arch}")

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    print('[zejun] ATOM prepare_model, model_arch = ', model_arch, ', model_cls = ', model_cls, flush=True)

    if is_vllm():
        _register_ops_to_vllm()
    elif is_sglang():
        _register_ops_to_sglang()

    from atom.plugin.plugin_config import generate_atom_config_in_plugin_mode
    atom_config = generate_atom_config_in_plugin_mode(config)

    # init aiter dist for using aiter custom collective ops
    _init_aiter_dist(config=atom_config)

    return model_cls(config=atom_config)
