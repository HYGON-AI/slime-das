# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
# from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from lmslim.layers.gemm.int8_utils import per_token_quant_int8
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import is_cuda, get_bool_env_var
from sglang.srt.layers.quantization.compressed_tensors import quant_ops as ops
_use_fused_rms_quant = get_bool_env_var("SGLANG_USE_FUSED_RMS_QUANT")
_use_fused_silu_mul_quant = get_bool_env_var("SGLANG_USE_FUSED_SILU_MUL_QUANT")

from lmslim import quant_ops 
_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import int8_scaled_mm
# TODO: remove vllm deps
from sglang.srt.utils import W8a8GetCacheJSON
W8A8_TRITONJSON=W8a8GetCacheJSON()

class CompressedTensorsW8A8Int8(CompressedTensorsScheme):

    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        self.w8a8_strategy=int(os.getenv('W8A8_SUPPORT_METHODS', '1'))  # TODO

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per channel
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight
            weight_scale = layer.weight_scale.data

            layer.weight = Parameter(weight.t(), requires_grad=False)
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            if self.input_symmetric:
                layer.input_scale = Parameter(
                    layer.input_scale.max(), requires_grad=False
                )
            else:
                input_scale = layer.input_scale
                input_zero_point = layer.input_zero_point

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)

                layer.input_scale = Parameter(scale, requires_grad=False)
                layer.input_zero_point = Parameter(azp, requires_grad=False)
        else:
            layer.input_scale = None
            layer.input_zero_point = None

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/cutlass_w8a8/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/8d59dbb00044a588cab96bcdc028006ed922eb06/csrc/quantization/cutlass_w8a8/Epilogues.md
        if not self.input_symmetric:
            weight = layer.weight
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = layer.input_zero_point * azp_adj
            layer.azp_adj = Parameter(azp_adj, requires_grad=False)
        else:
            layer.azp_adj = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader
            )
            layer.register_parameter("input_scale", input_scale)

            if not self.input_symmetric:
                # Note: compressed-tensors stores the zp using the same dtype
                # as the weights
                # AZP loaded as int8 but used as int32
                input_zero_point = PerTensorScaleParameter(
                    data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader
                )
                layer.register_parameter("input_zero_point", input_zero_point)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor],
        input_quant_args: Optional[list[torch.Tensor]] = None,
        silu_quant_args: Optional[list[torch.Tensor]] = None
    ) -> torch.Tensor:
        # TODO: add cutlass_scaled_mm_azp support
        if _use_fused_rms_quant and input_quant_args is not None:
            assert len(input_quant_args) == 2
            x_q, x_scale = input_quant_args
        elif _use_fused_silu_mul_quant and silu_quant_args is not None:
            x_q, x_scale = silu_quant_args
        else:
            x_q, x_scale = per_token_quant_int8(x)
        
        # return quant_ops.custom_scaled_mm(x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias)
        
        if self.w8a8_strategy==1:
            m=x_q.shape[0]
            k=x_q.shape[1]
            n=layer.weight.shape[1]
            
            if len(W8A8_TRITONJSON.triton_json_dict)==0:
                best_config=None
                
            elif f"1_{n}_{k}" in  W8A8_TRITONJSON.triton_json_dict:
                if m<=16:
                    m_=m
                elif m<=64:
                    m_= (m + 3) & -4 #取值到最近的4的倍数
                elif m<=160:
                    m_=(m + 7) & -8
                    
                elif m<200: #256
                    m_=160
                elif m<480: #512
                    m_=256
                elif m<960: #1024
                    m_=512
                elif m<2048:
                    m_=1024
                elif m<4096:
                    m_=2048
                elif m<6000:
                    m_=4096
                else:
                    m_=8192  

                best_config=W8A8_TRITONJSON.triton_json_dict[f"{m_}_{n}_{k}"]
                
            else: 
                best_config=None

            return ops.triton_scaled_mm(
                x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias, best_config=best_config
            )
        elif self.w8a8_strategy==2:
            return ops.cutlass_scaled_mm(x_q,
                                    layer.weight,
                                    scale_a=x_scale,
                                    scale_b=layer.weight_scale,
                                    out_dtype=x.dtype,
                                    bias=bias)
        elif self.w8a8_strategy==3:
            return ops.blaslt_scaled_mm(x_q,
                                    layer.weight,
                                    scale_a=x_scale,
                                    scale_b=layer.weight_scale,
                                    out_dtype=x.dtype,
                                    bias=None)
        else:
            return ops.rocblas_scaled_mm(x_q,
                                        layer.weight,
                                        scale_a=x_scale,
                                        scale_b=layer.weight_scale,
                                        out_dtype=x.dtype,
                                        bias=bias)
