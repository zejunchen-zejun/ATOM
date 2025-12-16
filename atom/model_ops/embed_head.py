# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import nn
import torch.nn.functional as F

from aiter.tuned_gemm import tgemm

from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.communication_op import tensor_model_parallel_all_gather

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as VllmQuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    VocabParallelEmbedding
)
from atom.model_ops.attention_mha import (
    _IS_PREFILL_FOR_PARALLEL_LMHEAD, 
    _CU_SEQLENS_Q_FOR_PARALLEL_LMHEAD
)


class ATOMVocabParallelEmbedding(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: VllmQuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim, prefix=prefix)
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # print('[zejun] ATOM ATOMVocabParallelEmbedding weight_loader', flush=True)
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)
        # print('[zejun] finish ATOM ATOMVocabParallelEmbedding weight_loader', flush=True)

    def forward(self, x: torch.Tensor):
        # print('[zejun] ATOM ATOMVocabParallelEmbedding forward', flush=True)
        if self.tp_size > 1:
            mask = torch.logical_and(x >= self.vocab_start_idx, x < self.vocab_end_idx)
            # mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y.masked_fill_(~mask.unsqueeze(1), 0)
            y = get_tp_group().all_reduce(y)
        return y


class ParallelLMHead(ATOMVocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            prefix=prefix,
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        print('[zejun] ATOM ParallelLMHead forward', flush=True)
        print('[zejun] ATOM ParallelLMHead forward, _IS_PREFILL_FOR_PARALLEL_LMHEAD = ', _IS_PREFILL_FOR_PARALLEL_LMHEAD, flush=True)
        print('[zejun] ATOM ParallelLMHead forward, _CU_SEQLENS_Q_FOR_PARALLEL_LMHEAD = ', _CU_SEQLENS_Q_FOR_PARALLEL_LMHEAD, flush=True)

        if _IS_PREFILL_FOR_PARALLEL_LMHEAD:
            print('[zejun] ATOM ParallelLMHead forward, last_indices = ', last_indices, flush=True)
            last_indices = _CU_SEQLENS_Q_FOR_PARALLEL_LMHEAD[1:] - 1
            x = x[last_indices].contiguous()
        logits = tgemm.mm(x, self.weight, self.bias)
        if self.tp_size > 1:
            logits = tensor_model_parallel_all_gather(logits)
        return logits
