import torch
from torch import nn
import torch.nn.functional as F

# import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group
from aiter import (
    QuantType,
    dtypes,
    gemm_a8w8_bpreshuffle,
    gemm_a8w8_blockscale,
    gemm_a4w4,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.tuned_gemm import tgemm
from typing import Optional, Callable
from functools import partial as functools_partial


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
        bias: bool = False,
        params_dtype=dtypes.bf16,
        quant_type=QuantType.No,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_group().rank
        self.tp_size = get_tp_group().world_size
        if tp_dim == 1:
            self.input_size = divide(input_size, self.tp_size)
        elif tp_dim == 0:
            self.output_size = divide(output_size, self.tp_size)
        self.weight = nn.Parameter(
            torch.empty((self.output_size, self.input_size), dtype=params_dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size, dtype=params_dtype))
            self.bias.weight_loader_process = self.weight_loader_process
        else:
            self.register_parameter("bias", None)
        self.quant_type = quant_type
        if quant_type != QuantType.No:
            if quant_type == QuantType.per_Tensor:
                self.weight.weight_loader_process = functools_partial(
                    self.weight_loader_process, post_process_func=shuffle_weight(16, 16)
                )
                self.weight_scale = nn.Parameter(torch.empty(1, dtype=dtypes.fp32))
            elif quant_type == QuantType.per_Token:
                self.weight.weight_loader_process = self.weight_loader_process
                self.weight_scale = nn.Parameter(
                    torch.empty(self.output_size, dtype=dtypes.fp32)
                )
            elif quant_type == QuantType.per_128x128:
                self.weight.weight_loader_process = self.weight_loader_process
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        divide(self.output_size, 128),
                        (self.input_size + 127) // 128,
                        dtype=dtypes.fp32,
                    )
                )
            elif quant_type == QuantType.per_1x32:
                self.weight.weight_loader_process = functools_partial(
                    self.weight_loader_process, post_process_func=shuffle_weight(16, 16)
                )
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        self.output_size,
                        (self.input_size + 31) // 32,
                        dtype=dtypes.fp8_e8m0,
                    )
                )
            self.weight_scale.weight_loader_process = self.weight_loader_process
        else:
            self.weight.weight_loader_process = self.weight_loader_process
            self.register_parameter("weight_scale", None)
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
        if self.weight_scale is not None:
            self.weight_scale.weight_loader = self.weight_loader

    @staticmethod
    def weight_loader_process(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        post_process_func: Callable = lambda a: a,
    ):
        param.data.copy_(post_process_func(loaded_weight))

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)

    def forward(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None, otype=dtypes.bf16
    ) -> torch.Tensor:
        if self.quant_type == QuantType.No:
            y = tgemm.mm(x, self.weight, self.bias)
        else:
            assert x_scale is not None
            if self.quant_type == QuantType.per_Tensor:
                y = tgemm.mm(
                    x,
                    self.weight,
                    self.bias,
                    otype=otype,
                    scale_a=x_scale,
                    scale_b=self.weight_scale,
                )
            elif self.quant_type == QuantType.per_Token:
                y = gemm_a8w8_bpreshuffle(
                    x, self.weight, x_scale, self.weight_scale, self.bias, dtype=otype
                )
            elif self.quant_type == QuantType.per_128x128:
                y = gemm_a8w8_blockscale(
                    x, self.weight, x_scale, self.weight_scale, self.bias, dtype=otype
                )
            elif self.quant_type == QuantType.per_1x32:
                y = gemm_a4w4(
                    x, self.weight, x_scale, self.weight_scale, self.bias, dtype=otype
                )
        if self.tp_dim == 1 and self.tp_size > 1:
            y = get_tp_group().all_reduce(y, open_fp8_quant=False)
        return y


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        params_dtype=dtypes.bf16,
        quant_type=QuantType.No,
    ):
        super().__init__(
            input_size,
            output_size,
            tp_dim=None,
            bias=bias,
            params_dtype=params_dtype,
            quant_type=quant_type,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        params_dtype=dtypes.bf16,
        quant_type=QuantType.No,
    ):
        self.tp_dim = 0
        super().__init__(
            input_size, output_size, self.tp_dim, bias, params_dtype, quant_type
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        params_dtype=dtypes.bf16,
        quant_type=QuantType.No,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, params_dtype, quant_type)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.weight_loader_process(param_data, loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        params_dtype=dtypes.bf16,
        quant_type=QuantType.No,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (
            self.total_num_heads + 2 * self.total_num_kv_heads
        ) * self.head_size
        super().__init__(input_size, output_size, bias, params_dtype, quant_type)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.weight_loader_process(param_data, loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        params_dtype=dtypes.bf16,
        quant_type=QuantType.No,
    ):
        self.tp_rank = get_tp_group().rank
        super().__init__(
            input_size,
            output_size,
            tp_dim=1,
            bias=bias if self.tp_rank == 0 else False,
            params_dtype=params_dtype,
            quant_type=quant_type,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)
