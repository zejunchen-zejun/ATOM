"""Runtime patches for SGLang TBO when using ATOM plugin models."""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_tbo_decode_metadata_patch() -> None:
    """Allow decode TBO split to handle SGLang padded batches.

    SGLang's TBO ``filter_batch`` expects every request-level metadata field to
    have ``batch_size`` entries. Decode batches can be padded for DP/shape
    alignment while ``ForwardBatch.rids`` only contains real request IDs. The
    model does not consume ``rids``; it is used for tracing/dumping. Patch
    ``filter_batch`` at plugin import time and temporarily pad ``rids`` only for
    the duration of child batch construction.
    """

    try:
        from sglang.srt.batch_overlap.two_batch_overlap import (
            TboForwardBatchPreparer,
        )
    except Exception:
        logger.exception("Failed to import SGLang TBO preparer for ATOM patch")
        return

    if getattr(TboForwardBatchPreparer, "_atom_decode_rids_patch", False):
        return

    original_filter_batch = TboForwardBatchPreparer.filter_batch.__func__

    def _patched_filter_batch(cls, batch, *args: Any, **kwargs: Any):
        original_rids = getattr(batch, "rids", None)
        padded_rids = _maybe_pad_decode_rids(batch, original_rids)
        if padded_rids is None:
            return original_filter_batch(cls, batch, *args, **kwargs)

        batch_for_filter = copy.copy(batch)
        batch_for_filter.rids = padded_rids
        return original_filter_batch(cls, batch_for_filter, *args, **kwargs)

    TboForwardBatchPreparer.filter_batch = classmethod(_patched_filter_batch)
    TboForwardBatchPreparer._atom_decode_rids_patch = True


def _maybe_pad_decode_rids(batch, rids):
    if rids is None:
        return None

    forward_mode = getattr(batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_decode():
        return None

    batch_size = getattr(batch, "batch_size", None)
    if batch_size is None or len(rids) == batch_size:
        return None

    if len(rids) > batch_size:
        return list(rids[:batch_size])

    # Preserve real request IDs and mark padded decode rows as dummy metadata.
    return list(rids) + [None] * (batch_size - len(rids))
