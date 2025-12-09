from .register import _ATOM_SUPPORTED_MODELS, _register_ops, _init_aiter_dist

# TODO: for sgl
def prepare_model(vllm_config):
    '''
    Prepare the model to upper framework, including
    register custom ops and init aiter dist
    '''
    model_arch = vllm_config.model_config.architectures[0]
    if model_arch not in _ATOM_SUPPORTED_MODELS:
        raise Warning(f"ATOM does not support the required model architecture: {model_arch}")

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    print('[zejun] ATOM prepare_model, model_arch = ', model_arch, ', model_cls = ', model_cls, flush=True)

    _register_ops()

    # TODO: init aiter dist
    _init_aiter_dist(vllm_config)

    # import os
    # pid = os.getpid()
    # from vllm.attention.backends.registry import _ATTN_OVERRIDES
    # for backend, name in _ATTN_OVERRIDES.items():
    #     print('[zejun][', pid, '] ATOM get_model, backend = ', backend, '. name = ', name, flush=True)

    return model_cls(vllm_config=vllm_config)
