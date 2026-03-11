import importlib.util
import importlib
import sys
import types

import pytest


# FIXME: remove it later when enabling fallback for unsupported models
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vllm is not installed in current test environment",
)
def test_vllm_wrapper_rejects_unsupported_model_arch(monkeypatch):
    # Avoid importing deep model-loader dependencies during test collection/import.
    fake_loader = types.ModuleType("atom.model_loader.loader")
    fake_loader.load_model_in_plugin_mode = lambda **kwargs: set()
    monkeypatch.setitem(sys.modules, "atom.model_loader.loader", fake_loader)

    model_wrapper = importlib.import_module("atom.plugin.vllm.model_wrapper")

    with pytest.raises(ValueError, match="not supported by ATOM OOT backend"):
        model_wrapper._get_atom_model_cls("UnknownModelForCausalLM")
