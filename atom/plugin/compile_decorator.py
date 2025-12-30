from typing import Type, TypeVar

T = TypeVar('T')


# TODO: align with atom support_torch_compile decorator
def support_torch_compile(**kwargs):
    """
    Decorator that conditionally applies torch.compile support to model classes based on the framework.

    - If using vLLM: applies vLLM's support_torch_compile decorator with parameters
    - If using SGLang: returns the class as-is (no-op decorator) for now
    """

    def compile_decorator(target_cls: Type[T]) -> Type[T]:
        """Inner decorator that wraps the class"""
        # Access the global variable directly to avoid partially initialized module issues
        from atom.utils.prepare import is_vllm, is_sglang
        if is_vllm():
            # Import and use vLLM's support_torch_compile decorator
            from vllm.compilation.decorators import support_torch_compile
            print('[zejun] ATOM support_torch_compile: using vllm support_torch_compile decorator', flush=True)
            return support_torch_compile(**kwargs)(target_cls)
        elif is_sglang():
            # For SGLang, return the class unchanged (no-op decorator)
            print('[zejun] ATOM support_torch_compile: for sglang, return the class undecorated', flush=True)
            return target_cls
        else:
            # TODO: support ATOM, for now, abort
            raise ValueError("For torch.compile, no framework is specified for plugin mode")

    return compile_decorator
