"""ATOM model wrappers for SGLang external model loading.

Registers model architecture classes via SGLANG_EXTERNAL_MODEL_PACKAGE,
replacing sglang's built-in implementations with ATOM-optimized versions.

To add a new model, append its architecture class name to _MODEL_NAMES.
"""

import copy

import logging
from contextvars import ContextVar
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger("atom.plugin.sglang.models")

# Context for patched DeepSeek attention layers that need wrapper state without
# changing every intermediate forward signature. ContextVar keeps nested or
# concurrent forwards isolated and lets us reliably restore the prior value.
_current_forward_batch: ContextVar[Optional[ForwardBatch]] = ContextVar(
    "atom_sglang_current_forward_batch", default=None
)


def get_current_forward_batch():
    return _current_forward_batch.get()


def _is_idle_forward(forward_batch: ForwardBatch) -> bool:
    forward_mode = getattr(forward_batch, "forward_mode", None)
    return bool(
        forward_mode is not None
        and hasattr(forward_mode, "is_idle")
        and forward_mode.is_idle()
    )


def _is_dummy_forward(forward_batch: ForwardBatch) -> bool:
    # SGLang's IDLE batch is the plugin-side equivalent of ATOM dummy run.
    return bool(getattr(forward_batch, "is_dummy_run", False) or _is_idle_forward(forward_batch))


def _get_forward_num_tokens(
    positions: Optional[torch.Tensor], forward_batch: ForwardBatch
) -> int:
    if positions is not None and hasattr(positions, "shape"):
        return int(positions.shape[0])
    batch_positions = getattr(forward_batch, "positions", None)
    if batch_positions is not None and hasattr(batch_positions, "shape"):
        return int(batch_positions.shape[0])
    return int(getattr(forward_batch, "seq_lens_sum", 0) or 0)

def _pad_dummy_like(
    tensor: Optional[torch.Tensor],
    *,
    length: int,
    fill_value: int | float = 0,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    shape = (length, *tensor.shape[1:])
    return torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)


def _materialize_atom_dummy_forward(
    input_ids: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    input_embeds: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    ForwardBatch,
    bool,
]:
    """Convert an empty SGLang IDLE batch into ATOM-style dummy forward inputs."""
    if not _is_dummy_forward(forward_batch) or _get_forward_num_tokens(positions, forward_batch) > 0:
        return input_ids, positions, input_embeds, forward_batch, False

    if positions is not None:
        dummy_positions = positions.new_zeros((1,))
    elif input_ids is not None:
        dummy_positions = torch.zeros((1,), dtype=torch.int64, device=input_ids.device)
    else:
        raise RuntimeError("Cannot materialize ATOM dummy forward without positions or input_ids")

    if input_ids is not None:
        dummy_input_ids = input_ids.new_zeros((1,))
    else:
        dummy_input_ids = None
    dummy_input_embeds = _pad_dummy_like(input_embeds, length=1, fill_value=0)

    model_forward_batch = copy.copy(forward_batch)
    model_forward_batch.positions = dummy_positions
    model_forward_batch.batch_size = 1
    model_forward_batch.seq_lens_sum = 1

    seq_lens = getattr(model_forward_batch, "seq_lens", None)
    if seq_lens is not None:
        if torch.is_tensor(seq_lens):
            model_forward_batch.seq_lens = seq_lens.new_ones((1,))
        else:
            model_forward_batch.seq_lens = [1]

    seq_lens_cpu = getattr(model_forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is not None:
        if torch.is_tensor(seq_lens_cpu):
            model_forward_batch.seq_lens_cpu = seq_lens_cpu.new_ones((1,))
        else:
            model_forward_batch.seq_lens_cpu = [1]

    return dummy_input_ids, dummy_positions, dummy_input_embeds, model_forward_batch, True


def _trim_hidden_states_for_output(hidden_states, num_tokens: int):
    if torch.is_tensor(hidden_states):
        return hidden_states[:num_tokens]
    if isinstance(hidden_states, tuple):
        return tuple(
            tensor[:num_tokens] if torch.is_tensor(tensor) else tensor
            for tensor in hidden_states
        )
    return hidden_states


def _align_dummy_num_tokens_across_dp(
    atom_config: Any,
    num_tokens: Optional[int],
    num_tokens_across_dp: Optional[torch.Tensor],
    is_dummy_run: bool,
):
    if (
        not is_dummy_run
        or num_tokens is None
        or num_tokens_across_dp is None
        or not hasattr(num_tokens_across_dp, "clone")
    ):
        return num_tokens_across_dp

    dp_rank = getattr(atom_config.parallel_config, "data_parallel_rank", None)
    if dp_rank is None:
        return num_tokens_across_dp
    if dp_rank < 0 or dp_rank >= num_tokens_across_dp.shape[0]:
        return num_tokens_across_dp
    if int(num_tokens_across_dp[dp_rank]) == int(num_tokens):
        return num_tokens_across_dp

    aligned = num_tokens_across_dp.clone()
    aligned[dp_rank] = int(num_tokens)
    return aligned


def _resolve_num_tokens_across_dp(
    atom_config: Any,
    forward_batch: ForwardBatch,
    num_tokens: Optional[int],
) -> Optional[torch.Tensor]:
    global_num_tokens_cpu = getattr(forward_batch, "global_num_tokens_cpu", None)
    if global_num_tokens_cpu is not None:
        return torch.tensor(global_num_tokens_cpu, dtype=torch.int32, device="cpu")

    if num_tokens is not None:
        dp_size = int(
            getattr(atom_config.parallel_config, "data_parallel_size", 1) or 1
        )
        return torch.full((dp_size,), int(num_tokens), dtype=torch.int32, device="cpu")

    global_num_tokens_gpu = getattr(forward_batch, "global_num_tokens_gpu", None)
    if global_num_tokens_gpu is not None:
        return global_num_tokens_gpu.to(device="cpu", dtype=torch.int32)

    return None


def _resolve_graph_bs(
    forward_batch: ForwardBatch,
    batch_size: int,
    num_tokens: Optional[int],
    num_tokens_across_dp: Optional[torch.Tensor],
) -> int:
    if num_tokens_across_dp is not None and getattr(num_tokens_across_dp, "numel", None):
        if num_tokens_across_dp.numel() > 0:
            return int(torch.max(num_tokens_across_dp).item())

    forward_mode = getattr(forward_batch, "forward_mode", None)
    is_prefill = bool(
        forward_mode is not None
        and hasattr(forward_mode, "is_prefill")
        and forward_mode.is_prefill()
    )
    if is_prefill and num_tokens is not None:
        return int(num_tokens)

    return int(batch_size)


def _set_sglang_forward_context(
    atom_config: Any,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
) -> None:
    """Bridge SGLang batch metadata into ATOM's global forward context."""
    from atom.utils.forward_context import AttentionMetaData, Context, set_forward_context

    forward_metadata = getattr(
        getattr(forward_batch, "attn_backend", None), "forward_metadata", None
    )
    max_seqlen_q = int(getattr(forward_metadata, "max_q_len", 0) or 0)
    if max_seqlen_q <= 0 and forward_batch.forward_mode.is_decode_or_idle():
        max_seqlen_q = 1

    attn_metadata = AttentionMetaData(max_seqlen_q=max_seqlen_q)
    batch_size = int(getattr(forward_batch, "batch_size", 0) or 0)
    is_dummy_run = _is_dummy_forward(forward_batch)
    is_prefill = forward_batch.forward_mode.is_prefill()

    num_tokens = None
    if positions is not None:
        num_tokens = int(positions.shape[0])
    elif getattr(forward_batch, "seq_lens_sum", None) is not None:
        num_tokens = int(forward_batch.seq_lens_sum)

    num_tokens_across_dp = _resolve_num_tokens_across_dp(
        atom_config, forward_batch, num_tokens
    )
    num_tokens_across_dp = _align_dummy_num_tokens_across_dp(
        atom_config,
        num_tokens,
        num_tokens_across_dp,
        is_dummy_run,
    )
    graph_bs = _resolve_graph_bs(
        forward_batch, batch_size, num_tokens, num_tokens_across_dp
    )
    context = Context(
        positions=positions,
        is_prefill=is_prefill,
        is_dummy_run=is_dummy_run,
        batch_size=batch_size,
        graph_bs=graph_bs,
    )
    set_forward_context(
        attn_metadata=attn_metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    )

def _reset_sglang_forward_context() -> None:
    from atom.utils.forward_context import reset_forward_context

    reset_forward_context()


_MODEL_NAMES = [
    "DeepseekV3ForCausalLM",
    "Qwen3MoeForCausalLM",
]

_DEEPSEEK_ARCHS = {
    "DeepseekV3ForCausalLM",
}


class _AtomCausalLMBaseForSglang(nn.Module):
    """Base ATOM model wrapper conforming to sglang's model interface.

    Delegates model creation and weight loading to ATOM's plugin system,
    while providing the forward signature and LogitsProcessorOutput return
    type that sglang expects.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Initializing ATOM backend for %s", self.__class__.__name__)

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        import atom

        # TODO: prepare_model() currently handles model construction, config
        # generation, attention backend registration, and distributed init.
        # Refactor so this wrapper only dispatches the attention backend
        # (register_ops_to_sglang + set_attn_cls), and let sglang handle
        # model construction directly
        self.model = atom.prepare_model(config=config, engine="sglang")
        if self.model is None:
            model_arch = getattr(config, "architectures", ["unknown"])[0]
            raise ValueError(
                f"ATOM failed to create model for architecture {model_arch}"
            )

        # Under SGLang dp-attention, ATOM runtime interprets non-MoE modules
        # like lm_head with tp=1 semantics, so plugin logits must not perform
        # an extra TP all-gather after local lm_head matmul.
        atom_config = getattr(self.model, "atom_config", None)
        if atom_config is None:
            atom_config = getattr(getattr(self.model, "model", None), "atom_config", None)
        if atom_config is None:
            from atom.config import get_current_atom_config

            atom_config = get_current_atom_config()
        if not hasattr(self.model, "atom_config"):
            self.model.atom_config = atom_config
        plugin_config = getattr(atom_config, "plugin_config", None)
        plugin_skip_all_gather = bool(
            getattr(plugin_config, "is_sglang", False)
            and (
                getattr(plugin_config, "sglang_enable_dp_attention", False)
                or getattr(atom_config, "enable_dp_attention", False)
            )
        )
        self.logits_processor = LogitsProcessor(
            config, skip_all_gather=plugin_skip_all_gather
        )

        # Apply ds model-specific sglang patches (attn dispatch, weight hooks, etc.)
        # TODO: will remove this after sglang supports atom attention backend
        arch = getattr(config, "architectures", [""])[0]
        self._uses_forward_batch_context = arch in _DEEPSEEK_ARCHS
        if arch in _DEEPSEEK_ARCHS:
            from atom.plugin.sglang.attention_backend.sgl_attention_mla import (
                setup_deepseek_for_sglang,
            )

            setup_deepseek_for_sglang(self.model)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **model_kwargs: Any,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        original_num_tokens = _get_forward_num_tokens(positions, forward_batch)
        (
            model_input_ids,
            model_positions,
            model_input_embeds,
            model_forward_batch,
            used_atom_dummy_materialization,
        ) = _materialize_atom_dummy_forward(
            input_ids,
            positions,
            input_embeds,
            forward_batch,
        )
        model_inputs = dict(
            input_ids=model_input_ids,
            positions=model_positions,
            intermediate_tensors=pp_proxy_tensors,
            inputs_embeds=model_input_embeds,
        )
        if self._uses_forward_batch_context:
            token = _current_forward_batch.set(model_forward_batch)
            try:
                _set_sglang_forward_context(
                    self.model.atom_config, model_forward_batch, model_positions
                )
                hidden_states = self.model(**model_inputs)
            finally:
                _reset_sglang_forward_context()
                _current_forward_batch.reset(token)
        else:
            hidden_states = self.model(
                **model_inputs,
                forward_batch=forward_batch,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
                **model_kwargs,
            )

        if self.pp_group.is_last_rank:
            if used_atom_dummy_materialization:
                hidden_states = _trim_hidden_states_for_output(
                    hidden_states, original_num_tokens
                )
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.model.lm_head,
                forward_batch,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The passed `weights` iterable from sglang is ignored because ATOM
        # uses its own weight loading pipeline (handling AITER-specific quant
        # formats, kv_b_proj splitting, etc.) that is incompatible with
        # sglang's default weight iterator.
        from atom.model_loader.loader import load_model_in_plugin_mode

        return load_model_in_plugin_mode(
            model=self.model, config=self.model.atom_config, prefix="model."
        )


EntryClass = []
for _name in _MODEL_NAMES:
    _cls = type(_name, (_AtomCausalLMBaseForSglang,), {})
    globals()[_name] = _cls
    EntryClass.append(_cls)
