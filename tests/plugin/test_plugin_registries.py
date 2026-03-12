import pytest

from atom.plugin import prepare as plugin_prepare
import atom.plugin.vllm.register as vllm_register


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


def test_register_platform_can_be_disabled(monkeypatch):
    monkeypatch.setattr(vllm_register, "disable_vllm_plugin", True, raising=False)
    assert vllm_register.register_platform() is None
