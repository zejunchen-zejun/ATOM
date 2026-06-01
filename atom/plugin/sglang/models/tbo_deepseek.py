"""SGLang TBO adapter for ATOM DeepSeek plugin models.

This module intentionally lives under ``atom.plugin`` so the first TBO
iterations do not require changing ATOM core modeling code. The initial
implementation validates the SGLang ``tbo_children`` path by running each child
ForwardBatch through the existing ATOM model and merging hidden states back to
the parent token order. Finer op-level overlap can be layered here with runtime
patches once split/merge correctness is established.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
_TBO_DEBUG = os.environ.get("ATOM_SGLANG_TBO_DEBUG", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def run_deepseek_tbo_if_available(
    *,
    model,
    model_inputs: dict[str, Any],
    atom_config,
    forward_batch,
    pp_proxy_tensors,
) -> tuple[bool, Any]:
    """Run plugin-side DeepSeek TBO if this batch is eligible.

    Returns ``(handled, output)``. ``handled=False`` means the caller should use
    the normal non-TBO forward path.
    """
    if not _can_run_plugin_tbo(forward_batch, pp_proxy_tensors):
        return False, None

    input_embeds = model_inputs.get("inputs_embeds")
    if input_embeds is not None:
        # Keep the first plugin step request-boundary safe. Input-embed TBO
        # needs the same child split/merge validation as PP tensors.
        _debug("skip: inputs_embeds TBO is not supported yet")
        return False, None

    _debug(
        "handle parent tokens=%s children=%s",
        _parent_num_tokens(forward_batch),
        [
            (
                getattr(child, "tbo_parent_token_range", None),
                getattr(child, "tbo_padded_len", None),
            )
            for child in forward_batch.tbo_children
        ],
    )

    outputs = []
    for child in forward_batch.tbo_children:
        child_inputs = _build_child_model_inputs(model_inputs, child)
        outputs.append(_run_child_forward(model, atom_config, child, child_inputs))

    return True, _merge_child_outputs(outputs, forward_batch)


def _can_run_plugin_tbo(forward_batch, pp_proxy_tensors) -> bool:
    if forward_batch is None or not getattr(forward_batch, "can_run_tbo", False):
        _debug(
            "skip: can_run_tbo is false (%s)",
            (
                None
                if forward_batch is None
                else getattr(forward_batch, "can_run_tbo", None)
            ),
        )
        return False
    children = getattr(forward_batch, "tbo_children", None)
    if children is None or len(children) != 2:
        _debug(
            "skip: expected two tbo_children, got %s",
            None if children is None else len(children),
        )
        return False
    if pp_proxy_tensors is not None:
        # TODO: split PPProxyTensors / IntermediateTensors per child.
        _debug("skip: PP proxy tensors TBO is not supported yet")
        return False
    has_ranges = all(
        getattr(child, "tbo_parent_token_range", None) is not None
        for child in children
    )
    if not has_ranges:
        _debug("skip: one or more children miss tbo_parent_token_range")
        return False
    for child in children:
        start, end = child.tbo_parent_token_range
        padded_len = getattr(child, "tbo_padded_len", None)
        if end <= start or padded_len == 0:
            _debug(
                "skip: empty child range=%s padded_len=%s",
                child.tbo_parent_token_range,
                padded_len,
            )
            return False
    return True


def _debug(message: str, *args):
    if _TBO_DEBUG:
        logger.warning("[ATOM SGLang TBO] " + message, *args)


def _build_child_model_inputs(parent_inputs: dict[str, Any], child) -> dict[str, Any]:
    input_ids = getattr(child, "input_ids", None)
    positions = getattr(child, "positions", None)

    if input_ids is None:
        input_ids = _slice_parent_token_tensor(parent_inputs.get("input_ids"), child)
    if positions is None:
        positions = _slice_parent_token_tensor(parent_inputs.get("positions"), child)

    return {
        "input_ids": input_ids,
        "positions": positions,
        "intermediate_tensors": None,
        "inputs_embeds": None,
    }


def _slice_parent_token_tensor(tensor: Optional[torch.Tensor], child):
    if tensor is None:
        return None
    start, end = child.tbo_parent_token_range
    sliced = tensor[slice(start, end)]
    return _pad_token_dim(sliced, child.tbo_padded_len)


def _pad_token_dim(tensor: torch.Tensor, padded_len: Optional[int]):
    if padded_len is None or tensor.shape[0] == padded_len:
        return tensor
    output = torch.zeros(
        (padded_len, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    output[: tensor.shape[0]] = tensor
    return output


def _run_child_forward(model, atom_config, child_batch, child_inputs: dict[str, Any]):
    from atom.plugin.sglang.models.base_model_wrapper import (
        SGLangForwardBatchMetadata,
        _reset_sglang_forward_context,
        _set_sglang_forward_context,
    )

    metadata = SGLangForwardBatchMetadata.build(child_batch)
    with SGLangForwardBatchMetadata.bind(metadata):
        try:
            _set_sglang_forward_context(
                atom_config, child_batch, child_inputs["positions"]
            )
            return model(**child_inputs)
        finally:
            _reset_sglang_forward_context()


def _merge_child_outputs(outputs: list[Any], parent_batch):
    if all(torch.is_tensor(output) for output in outputs):
        return _merge_child_tensors(outputs, parent_batch)

    if all(isinstance(output, tuple) for output in outputs):
        merged_hidden = _merge_child_tensors(
            [output[0] for output in outputs], parent_batch
        )
        if len(outputs[0]) == 1 or outputs[0][1] is None:
            return (merged_hidden,)
        # Preserve auxiliary outputs only when every child returns the same count.
        aux_outputs = []
        for aux_idx in range(len(outputs[0][1])):
            aux_outputs.append(
                _merge_child_tensors(
                    [output[1][aux_idx] for output in outputs], parent_batch
                )
            )
        return merged_hidden, aux_outputs

    output_types = [type(output) for output in outputs]
    raise TypeError(f"Unsupported DeepSeek TBO child output types: {output_types}")


def _merge_child_tensors(tensors: list[torch.Tensor], parent_batch):
    original_len = _parent_num_tokens(parent_batch)
    first = tensors[0]
    merged = torch.zeros(
        (original_len, *first.shape[1:]), dtype=first.dtype, device=first.device
    )
    for tensor, child in zip(tensors, parent_batch.tbo_children, strict=True):
        start, end = child.tbo_parent_token_range
        merged[start:end] = tensor[: end - start]
    return merged


def _parent_num_tokens(parent_batch) -> int:
    input_ids = getattr(parent_batch, "input_ids", None)
    if input_ids is not None:
        return int(input_ids.shape[0])
    num_token_non_padded = getattr(parent_batch, "num_token_non_padded", None)
    if torch.is_tensor(num_token_non_padded):
        return int(num_token_non_padded.item())
    if num_token_non_padded is not None:
        return int(num_token_non_padded)
    return max(child.tbo_parent_token_range[1] for child in parent_batch.tbo_children)
