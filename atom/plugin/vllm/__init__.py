"""vLLM plugin integration for ATOM."""

from .register import patch_model_registry, register_platform

__all__ = ["register_platform", "patch_model_registry"]
