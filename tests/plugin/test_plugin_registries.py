import sys
import types
import importlib
import importlib.util

import pytest

from atom.plugin import prepare as plugin_prepare
import atom.plugin.vllm.register as vllm_register


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vllm is not installed in current test environment",
)
def test_register_platform_returns_oot_platform(monkeypatch):
    rocm_module = types.ModuleType("vllm.platforms.rocm")

    class _RocmPlatform:
        pass

    rocm_module.RocmPlatform = _RocmPlatform
    vllm_platforms = types.ModuleType("vllm.platforms")
    vllm_platforms.current_platform = None

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", vllm_platforms)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm_module)

    monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN", "0")
    monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", "0")

    import atom.plugin.vllm.platform as platform_module

    importlib.reload(platform_module)
    importlib.reload(vllm_register)

    platform_path = vllm_register.register_platform()
    module_name, class_name = platform_path.rsplit(".", 1)
    vllm_platforms.current_platform = getattr(
        importlib.import_module(module_name), class_name
    )

    # get current platform from vllm side and validate it is ATOM platform.
    assert vllm_platforms.current_platform is platform_module.ATOMPlatform


def test_register_platform_can_be_disabled(monkeypatch):
    monkeypatch.setattr(vllm_register, "disable_vllm_plugin", True, raising=False)
    assert vllm_register.register_platform() is None
