import importlib
import importlib.util
import sys
import types

import pytest


def test_disable_vllm_plugin_flag_disables_platform(monkeypatch):
    # ATOM_DISABLE_VLLM_PLUGIN takes precedence:
    # when it is 1, vLLM should not get ATOM platform/attention at all.
    for disable_attention in ("0", "1"):
        monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN", "1")
        monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", disable_attention)

        import atom.plugin.vllm.platform as platform_module
        import atom.plugin.vllm.register as register_module

        importlib.reload(platform_module)
        importlib.reload(register_module)

        assert platform_module.ATOMPlatform is None
        assert register_module.register_platform() is None


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vllm is not installed in current test environment",
)
def test_disable_vllm_plugin_attention_fallbacks_to_non_atom_backend(monkeypatch):
    rocm_module = types.ModuleType("vllm.platforms.rocm")

    class _RocmPlatform:
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config):
            return "vllm.default.backend"

    rocm_module.RocmPlatform = _RocmPlatform

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", types.ModuleType("vllm.platforms")
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm_module)
    monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN", "0")
    monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", "1")

    import atom.plugin.vllm.platform as platform_module

    importlib.reload(platform_module)

    result = platform_module.ATOMPlatform.get_attn_backend_cls(
        selected_backend="x",
        attn_selector_config=types.SimpleNamespace(use_mla=True),
    )
    assert result == "vllm.default.backend"
