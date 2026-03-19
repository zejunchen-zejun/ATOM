"""ATOM external model package for upstream SGLang.

Loaded via SGLANG_EXTERNAL_MODEL_PACKAGE=atom.sglang_ext.
Provides ATOM-optimized model implementations for AMD GPUs,
replacing sglang's built-in versions via per-architecture EntryClass
registration (one submodule per model family):

  - deepseek.py  -> DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
  - qwen.py      -> Qwen3MoeForCausalLM, Qwen3ForCausalLM

DeepSeek-specific monkey-patches (MLA -> MHA, asymmetric K/V dims) are
applied at import time but guarded by architecture checks and the
_atom_mha_override flag, so they are no-ops for non-DeepSeek models.

Timing: the external package is imported when registry.py runs during model
loading.  However, ModelConfig is already instantiated with MLA settings at
that point.  So we patch init_memory_pool (which runs AFTER model loading) to
correct runner.use_mla_backend, model_config.attention_arch, head_dim, and
v_head_dim in-place before memory profiling and KV pool allocation.
"""

import logging

logger = logging.getLogger("atom.plugin.sglang.oot")

_ATOM_DEEPSEEK_ARCHS = {
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
}

# Filled by the model wrapper's __init__ with (head_dim, v_head_dim)
_atom_mha_override = None


def _set_mha_override(head_dim: int, v_head_dim: int):
    """Called by the model wrapper to signal MHA override is needed."""
    global _atom_mha_override
    _atom_mha_override = (head_dim, v_head_dim)


def _patch_model_config():
    """Patch _derive_model_shapes for any ModelConfig created AFTER this runs.

    This covers ATOM's internal ModelConfig.from_server_args call inside
    generate_atom_config_for_plugin_mode.  It does NOT help the scheduler's
    ModelConfig which is already created before registry.py is imported.
    """
    from sglang.srt.configs.model_config import AttentionArch, ModelConfig

    _original = ModelConfig._derive_model_shapes

    def _patched(self):
        _original(self)
        archs = getattr(self.hf_config, "architectures", [])
        if any(a in _ATOM_DEEPSEEK_ARCHS for a in archs):
            self.head_dim = (
                self.hf_text_config.qk_nope_head_dim
                + self.hf_text_config.qk_rope_head_dim
            )
            self.v_head_dim = self.hf_text_config.v_head_dim
            self.attention_arch = AttentionArch.MHA
            logger.info(
                "ATOM: _derive_model_shapes override -> MHA, "
                "head_dim=%d, v_head_dim=%d",
                self.head_dim,
                self.v_head_dim,
            )

    ModelConfig._derive_model_shapes = _patched
    logger.info("ATOM: Patched ModelConfig._derive_model_shapes")


def _patch_init_memory_pool():
    """Patch init_memory_pool on ModelRunnerKVCacheMixin.

    This runs AFTER model loading (where the ATOM model sets _atom_mha_override)
    but BEFORE memory profiling (get_cell_size_per_token), KV pool allocation
    (_init_pools), and attention backend creation (init_attention_backend).

    It corrects the already-created ModelConfig instance and runner state:
      - model_config.attention_arch  (MLA -> MHA)
      - model_config.head_dim / v_head_dim
      - runner.use_mla_backend       (True -> False)
      - global_server_args.use_mla_backend
    """
    try:
        import sglang.srt.model_executor.model_runner_kv_cache_mixin as _mixin_mod
    except ImportError:
        logger.warning("ATOM: Could not import model_runner_kv_cache_mixin")
        return

    _original = _mixin_mod.ModelRunnerKVCacheMixin.init_memory_pool

    def _patched(self, pre_model_load_memory):
        if _atom_mha_override is not None:
            from sglang.srt.configs.model_config import AttentionArch
            from sglang.srt.server_args import get_global_server_args

            head_dim, v_head_dim = _atom_mha_override
            self.model_config.attention_arch = AttentionArch.MHA
            self.model_config.head_dim = head_dim
            self.model_config.v_head_dim = v_head_dim
            self.use_mla_backend = False
            try:
                get_global_server_args().use_mla_backend = False
            except Exception:
                pass
            logger.info(
                "ATOM: Fixed runner state -> MHA, "
                "head_dim=%d, v_head_dim=%d, use_mla_backend=False",
                head_dim,
                v_head_dim,
            )
        _original(self, pre_model_load_memory)

    _mixin_mod.ModelRunnerKVCacheMixin.init_memory_pool = _patched
    logger.info("ATOM: Patched ModelRunnerKVCacheMixin.init_memory_pool")


def _patch_kv_pool():
    """Inject v_head_dim into MHATokenToKVPool creation.

    Upstream sglang doesn't pass v_head_dim when creating MHATokenToKVPool.
    DeepSeek has asymmetric K/V dims (K=192, V=128), so the V buffer must
    use the correct dimension.
    """
    try:
        import sglang.srt.mem_cache.memory_pool as _mem_pool_mod
        import sglang.srt.model_executor.model_runner_kv_cache_mixin as _kv_mixin_mod
    except ImportError:
        logger.warning(
            "ATOM: Could not import KV pool modules. "
            "Asymmetric K/V head dims may not work correctly."
        )
        return

    _OriginalMHAPool = _mem_pool_mod.MHATokenToKVPool

    class _PatchedMHATokenToKVPool(_OriginalMHAPool):
        def __init__(self, *args, v_head_dim=None, **kwargs):
            if v_head_dim is None and _atom_mha_override is not None:
                v_head_dim = _atom_mha_override[1]
            super().__init__(*args, v_head_dim=v_head_dim, **kwargs)

    _kv_mixin_mod.MHATokenToKVPool = _PatchedMHATokenToKVPool
    logger.info("ATOM: Patched MHATokenToKVPool for asymmetric K/V head dims")


_patch_model_config()
_patch_init_memory_pool()
_patch_kv_pool()
