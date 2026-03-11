import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vllm is not installed in current test environment",
)
def test_vllm_import_paths_guardrail():
    """Guardrail for OOT vLLM import paths used by ATOM plugin mode."""
    # attention.py / paged_attention.py (new path with legacy fallback)
    try:
        from vllm.attention.layer import Attention, MLAAttention, AttentionType
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention
        from vllm.v1.attention.backend import AttentionType

    # attention.py
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config,
        get_layers_from_vllm_config,
    )
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
        QueryLenSupport,
    )
    from vllm.utils.math_utils import cdiv, round_down
    from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
    from vllm.v1.attention.backends.utils import (
        get_dcp_local_seq_lens,
        split_decodes_and_prefills,
        split_decodes_prefills_and_extends,
    )
    from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

    # model_wrapper.py (core vLLM model interfaces)
    from vllm.model_executor.models.interfaces import SupportsPP, SupportsQuant
    from vllm.model_executor.models.interfaces_base import (
        VllmModel,
        VllmModelForTextGeneration,
    )
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.sequence import IntermediateTensors

    # attention_mla.py / platform.py / register.py
    from vllm import _custom_ops
    from vllm.distributed.parallel_state import get_dcp_group
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import RocmPlatform

    assert all(
        obj is not None
        for obj in [
            Attention,
            MLAAttention,
            AttentionType,
            QueryLenSupport,
            MLACommonMetadataBuilder,
            cdiv,
            round_down,
            AttentionCGSupport,
            AttentionMetadataBuilder,
            get_dcp_local_seq_lens,
            split_decodes_and_prefills,
            split_decodes_prefills_and_extends,
            cp_lse_ag_out_rs,
            merge_attn_states,
            VllmConfig,
            get_current_vllm_config,
            get_layers_from_vllm_config,
            SupportsPP,
            SupportsQuant,
            VllmModel,
            VllmModelForTextGeneration,
            ModelRegistry,
            IntermediateTensors,
            _custom_ops,
            get_dcp_group,
            current_platform,
            RocmPlatform,
        ]
    )
