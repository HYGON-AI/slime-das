from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from collections import defaultdict
from sglang.srt.distributed import get_moe_expert_parallel_rank, get_moe_expert_parallel_world_size
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_marlin import SlimQuantCompressedTensorsMarlinConfig
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import SlimQuantW4A8Int8MarlinConfig
import torch

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4DynamicMoEMethod,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    # should_use_flashinfer_trtllm_moe, # 找不到
)
from sglang.srt.layers.moe.ep_moe.kernels import (
    ep_gather,
    ep_scatter,
    silu_and_mul_masked_post_quant_fwd,
    tma_align_input_scale,
    per_token_quant_int8_triton_opt,
    build_m_indices_triton,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFusedMoE, FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.utils import ceil_div, dispose_tensor, get_bool_env_var, is_hip, is_npu, direct_register_custom_op
from sglang.srt.utils.offloader import get_offloader

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )
    
from deepgemm import m_grouped_w4a8_gemm_nt_masked, m_grouped_w8a8_gemm_nt_masked, m_grouped_i8_gemm_nt_contiguous
from lightop import fuse_silu_mul_quant_ep, fuse_silu_mul_quant
from lmslim.layers.gemm.int8_utils import per_token_quant_int8

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
elif _is_npu:
    import torch_npu


logger = logging.getLogger(__name__)

#------ custom op for lightop
def m_grouped_w4a8_gemm_nt_masked_wrapper(
    a0: torch.Tensor, a1: torch.Tensor,
    b0: torch.Tensor, b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int
    ) -> torch.Tensor:
    return m_grouped_w4a8_gemm_nt_masked(
            (a0, a1),
            (b0, b1),
            d,
            masked_m,
            expected_m_per_group,
            config={"MODE": 1000,}
        )
def m_grouped_w4a8_gemm_nt_masked_fake(
    a0: torch.Tensor, a1: torch.Tensor,
    b0: torch.Tensor, b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int
    ) -> torch.Tensor:
    return d
def m_grouped_w8a8_gemm_nt_masked_wrapper(
    a0: torch.Tensor, a1: torch.Tensor,
    b0: torch.Tensor, b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int
    ) -> torch.Tensor:
    return m_grouped_w8a8_gemm_nt_masked(
            (a0, a1),
            (b0, b1),
            d,
            masked_m,
            expected_m_per_group,
            config={"MODE": 1000,}
        )
def m_grouped_w8a8_gemm_nt_masked_fake(
    a0: torch.Tensor, a1: torch.Tensor,
    b0: torch.Tensor, b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int
    ) -> torch.Tensor:
    return d

def fuse_silu_mul_quant_ep_wrapper(
    input: torch.Tensor,
    tokens_per_expert: Optional[torch.Tensor] = None,
    num_local_tokens_tensor: Optional[torch.Tensor] = None,
    topk:int=1,
    expect_m:int=-1) -> tuple[torch.Tensor, torch.Tensor]:
    return fuse_silu_mul_quant_ep(
        input,
        tokens_per_expert,
        num_local_tokens_tensor,
        topk,
        expect_m
    )


def fuse_silu_mul_quant_ep_fake(
    input: torch.Tensor,
    tokens_per_expert: Optional[torch.Tensor] = None,
    num_local_tokens_tensor: Optional[torch.Tensor] = None,
    topk:int=1,
    expect_m:int=-1) -> tuple[torch.Tensor, torch.Tensor]:
    E, T, H = input.shape
    d = H // 2
    output = torch.empty(E, T, d, dtype=torch.int8, device=input.device)
    scales = torch.empty((E, T, 1),
                        device=input.device,
                        dtype=torch.float32)
    return output, scales


direct_register_custom_op(
    op_name="m_grouped_w4a8_gemm_nt_masked",
    op_func=m_grouped_w4a8_gemm_nt_masked_wrapper,
    mutates_args=[],
    fake_impl=m_grouped_w4a8_gemm_nt_masked_fake
)
direct_register_custom_op(
    op_name="m_grouped_w8a8_gemm_nt_masked",
    op_func=m_grouped_w8a8_gemm_nt_masked_wrapper,
    mutates_args=[],
    fake_impl=m_grouped_w8a8_gemm_nt_masked_fake
)
direct_register_custom_op(
    op_name="fuse_silu_mul_quant_ep",
    op_func=fuse_silu_mul_quant_ep_wrapper,
    mutates_args=[],
    fake_impl=fuse_silu_mul_quant_ep_fake
)

#------


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        if _use_aiter or _is_npu:
            self.deprecate_flag = False
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
        elif isinstance(quant_config, SlimQuantW4A8Int8MarlinConfig):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.activation_scheme = None
            self.use_w4a8_marlin = True
            self.use_w8a8_marlin = False
        elif isinstance(quant_config, SlimQuantCompressedTensorsMarlinConfig):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.activation_scheme = None
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = True  
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4afp8 = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False

        self.deepep_mode = get_deepep_mode()

        # if self.deepep_mode.enable_low_latency() and not _is_npu:
        #     # NPU supports low_latency deepep without deepgemm
        #     assert (
        #         deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        #     ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        if is_in_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return torch.ops.sglang.moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        # TODO: can we call super().forward here?
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        hidden_states = self.dispatcher.combine(
            combine_input=combine_input,
        )

        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):

        if self.deprecate_flag:
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            output = self.forward_aiter(dispatch_output)
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)
        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            #assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8:
                output = self.forward_deepgemm_contiguous(dispatch_output)
            elif self.use_w4a8_marlin:
                output = self.forward_deepgemm_w4a8_marlin_contiguous(dispatch_output)
            elif self.use_w8a8_marlin:
                output = self.forward_groupgemm_w8a8_marlin_contiguous(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                raise ValueError(
                    f"Dispatch output is not supported"
                )
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):

            if (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            elif self.use_w4a8_marlin:
                output = self.forward_groupgemm_w4a8_marlin_masked(dispatch_output)
            elif self.use_w8a8_marlin:
                output = self.forward_groupgemm_w8a8_marlin_masked(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )
        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        hidden_states, topk_ids, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_ids_copy = topk_ids.to(torch.int32)
        topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_deepgemm_w4a8_marlin_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        hidden_states, hidden_states_scale, topk_idx, topk_weights, num_recv_tokens_per_expert = (
            dispatch_output
        )
        #hidden_states_int8, hidden_states_scale = hidden_states_int8
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        all_tokens = sum(num_recv_tokens_per_expert)

        if all_tokens <= 0:
            return hidden_states.bfloat16()
        rank_expert_offset = get_moe_expert_parallel_rank() * (self.num_experts // get_moe_expert_parallel_world_size())
        topk_idx = torch.where(
            topk_idx == -1,
            self.num_experts - 1 if rank_expert_offset == 0 else 0,
            topk_idx + rank_expert_offset)
        expert_output = self.quant_method.apply_ep(
            x=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_ids=topk_idx,
            topk_weights=topk_weights,
            global_num_experts=self.moe_runner_config.num_experts,
            expert_map=self.expert_map,
            activation=self.moe_runner_config.activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            use_nn_moe=False,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_scale=hidden_states_scale,
            routed_scaling_factor = self.moe_runner_config.routed_scaling_factor,
        )
        return expert_output

    # def forward_groupgemm_w8a8_marlin_contiguous(
    #     self,
    #     dispatch_output: DeepEPNormalOutput,
    # ):
    #     hidden_states, hidden_states_scale, topk_idx, topk_weights, num_recv_tokens_per_expert = dispatch_output
    #
    #     assert self.quant_method is not None
    #     assert self.moe_runner_config.activation == "silu"
    #     all_tokens = sum(num_recv_tokens_per_expert)
    #     if all_tokens <= 0:
    #         return hidden_states.bfloat16()
    #
    #     device = hidden_states.device
    #     M = hidden_states.shape[0]
    #     K = hidden_states.shape[1]
    #     topk = topk_idx.shape[1]
    #
    #     active_experts = set()
    #     token_expert_pos = [None] * M
    #     for t in range(M):
    #         lst = []
    #         for pos in range(topk):
    #             e = int(topk_idx[t, pos].item())
    #             if e >= 0:
    #                 lst.append((e, pos))
    #                 active_experts.add(e)
    #         token_expert_pos[t] = lst
    #
    #     if not active_experts:
    #         return hidden_states.bfloat16()
    #     active_experts = sorted(list(active_experts))
    #
    #     counts = defaultdict(int)
    #     for t in range(M):
    #         for (e, pos) in token_expert_pos[t]:
    #             counts[e] += 1
    #
    #     per_expert_block = {}
    #     for e in active_experts:
    #         cnt = counts[e]
    #         needed = ((cnt + 255) // 256) * 256  # same as ceil(cnt/256)*256
    #         per_expert_block[e] = max(256, needed)
    #
    #     expert_slot_offset = {}
    #     offset = 0
    #     for e in active_experts:
    #         expert_slot_offset[e] = offset
    #         offset += per_expert_block[e]
    #     pad_M = offset
    #
    #     hidden_states_packed = torch.empty((pad_M, K), device=device, dtype=hidden_states.dtype)
    #     hidden_states_scale_packed = torch.empty((pad_M,), device=device, dtype=hidden_states_scale.dtype)
    #     m_indices = torch.full((pad_M,), -1, device=device, dtype=torch.int32)
    #
    #     slot_counters = {e: 0 for e in active_experts}
    #     token_row_weight_list = {t: [] for t in range(M)}
    #
    #     for t in range(M):
    #         for (e, pos) in token_expert_pos[t]:
    #             start = expert_slot_offset[e]
    #             slot = slot_counters[e]
    #             row = start + slot
    #             hidden_states_packed[row] = hidden_states[t]
    #             hidden_states_scale_packed[row] = hidden_states_scale[t]
    #             m_indices[row] = e
    #             slot_counters[e] += 1
    #
    #             # record weight (as float32 on device)
    #             w = topk_weights[t, pos]
    #             w_f = w.float() if w.dtype != torch.float32 else w
    #             token_row_weight_list[t].append((row, w_f))
    #
    #     # q_a1_all, q_a1_scale = per_token_quant_int8(hidden_states_packed)
    #     N = self.w13_weight.size(1)
    #     gateup_output = torch.empty((pad_M, N * 16), device=device, dtype=torch.bfloat16)
    #     m_grouped_w8a8_gemm_nt_contig_asm(
    #         (hidden_states_packed, hidden_states_scale_packed),
    #         (self.w13_weight, self.w13_weight_scale),
    #         gateup_output,
    #         m_indices,
    #     )
    #     del hidden_states_packed, hidden_states_scale_packed
    #     q_a2_all, q_a2_scale = fuse_silu_mul_quant(gateup_output)
    #     down_output = torch.empty((pad_M, K), device=device, dtype=torch.bfloat16)
    #     down_output = m_grouped_w8a8_gemm_nt_contig_asm(
    #         (q_a2_all, q_a2_scale),
    #         (self.w2_weight, self.w2_weight_scale),
    #         down_output,
    #         m_indices,
    #     )
    #     result = torch.zeros((M, K), device=device, dtype=down_output.dtype)
    #     for t in range(M):
    #         for (row, w) in token_row_weight_list[t]:
    #             result[t].addcmul_(down_output[row].float(), w)
    #
    #     return result.to(down_output.dtype)

    def forward_groupgemm_w8a8_marlin_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output

        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()

        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()

        M, K = hidden_states.size()
        N = self.w13_weight.size(1)
        w13_weight_int8 = (
            self.w13_weight,
            (self.w13_weight_scale),
        )
        w2_weight_int8 = (
            self.w2_weight,
            (self.w2_weight_scale),
        )

        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype
        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            (
                torch.empty(
                    (all_tokens, hidden_states_scale.shape[-1]),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            ),
        ]
        output_index = torch.full_like(topk_ids, -1)

        if get_offloader().forbid_copy_engine_usage:
            num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
                num_recv_tokens_per_expert
            )
        else:
            num_recv_tokens_per_expert_gpu = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int32,
                pin_memory=True,
                device="cpu",
            ).cuda(non_blocking=True)
        expert_start_loc = torch.zeros_like(num_recv_tokens_per_expert_gpu)
        local_num_expert = num_recv_tokens_per_expert_gpu.shape[0]
        m_indices = build_m_indices_triton(topk_ids, hidden_states.device, local_num_expert)

        ep_scatter(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
        )

        gateup_output = torch.zeros(
            (all_tokens, N * 16),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        m_grouped_i8_gemm_nt_contiguous(
            input_tensor,
            w13_weight_int8,
            gateup_output,
            m_indices,
        )
        del input_tensor

        q_a2_all, q_a2_scale = fuse_silu_mul_quant(gateup_output)
        del gateup_output

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        m_grouped_i8_gemm_nt_contiguous(
            (q_a2_all, q_a2_scale),
            w2_weight_int8,
            down_output,
            m_indices,
        )

        gather_out = torch.zeros(
            hidden_states_shape,
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        ep_gather(down_output, topk_ids, topk_weights, output_index, gather_out)
        del down_output

        return gather_out


    def forward_deepgemm_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()
        M, K = hidden_states.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

        w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        w2_weight_fp8 = (
            self.w2_weight,
            (
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
        )

        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            (
                # TODO check whether need `zeros`
                torch.zeros(
                    (ceil_div(K // 128, 4), all_tokens),
                    device=hidden_states.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else torch.empty(
                    (all_tokens, K // 128),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states.device, dtype=torch.int32
        )
        output_index = torch.empty_like(topk_ids)

        if get_offloader().forbid_copy_engine_usage:
            num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
                num_recv_tokens_per_expert
            )
        else:
            num_recv_tokens_per_expert_gpu = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int32,
                pin_memory=True,
                device="cpu",
            ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

        ep_scatter(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        dispose_tensor(hidden_states)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            input_tensor, w13_weight_fp8, gateup_output, m_indices
        )
        del input_tensor
        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input,
            scale_block_size,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del down_input
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = tma_align_input_scale(down_input_scale)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
        )
        del down_input_fp8, down_input_scale

        gather_out = torch.empty(
            hidden_states_shape,
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_ids, topk_weights, output_index, gather_out)

        return gather_out

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
        )
        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_groupgemm_w4a8_marlin_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):

        hidden_states, hidden_states_scale, _, _, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        # base shapes
        num_groups, m, k = hidden_states.size()
        expected_m = min(m, expected_m)

        # ---- first quant: ensure float input for quantizer ----
        # q_a1_all, q_a1_scale = per_token_quant_int8_triton_opt(hidden_states, masked_m)
        # ---- weights & scales ----
        w13_weight = self.w13_weight
        w13_scales = self.w13_weight_scale
        w2_weight = self.w2_weight
        w2_scales = self.w2_weight_scale

        n1 = w13_scales.size(1)
        gateup_output = torch.empty((num_groups, m, n1), device=hidden_states.device, dtype=torch.bfloat16)

        # ---- first GEMM ----
        torch.ops.sglang.m_grouped_w4a8_gemm_nt_masked(
            hidden_states, hidden_states_scale,
            w13_weight, w13_scales,
            gateup_output,
            masked_m,
            expected_m,
        )

        q_a2_all, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(gateup_output, masked_m)

        # ---- second GEMM ----
        n2 = w2_scales.size(1)
        down_output = torch.empty((num_groups, m, n2), device=q_a2_all.device, dtype=torch.bfloat16)

        torch.ops.sglang.m_grouped_w4a8_gemm_nt_masked(
            q_a2_all, q_a2_scale,
            w2_weight, w2_scales,
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    def forward_groupgemm_w8a8_marlin_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):

        hidden_states, hidden_states_scale, topk_ids, _, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        # base shapes
        num_groups, m, k = hidden_states.size()
        expected_m = min(m, expected_m)

        # ---- first quant: ensure float input for quantizer ----
        # q_a1_all, q_a1_scale = per_token_quant_int8_triton_opt(hidden_states, masked_m)

        # ---- weights & scales ----
        w13_weight = self.w13_weight
        w13_scales = self.w13_weight_scale
        w2_weight = self.w2_weight
        w2_scales = self.w2_weight_scale

        n1 = w13_scales.size(1)
        gateup_output = torch.empty((num_groups, m, n1), device=hidden_states.device, dtype=torch.bfloat16)

        # ---- first GEMM ----
        torch.ops.sglang.m_grouped_w8a8_gemm_nt_masked(
            hidden_states, hidden_states_scale,
            w13_weight, w13_scales,
            gateup_output,
            masked_m,
            expected_m,
        )

        q_a2_all, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(gateup_output, masked_m)

        # ---- second GEMM ----
        n2 = w2_scales.size(1)
        down_output = torch.empty((num_groups, m, n2), device=q_a2_all.device, dtype=torch.bfloat16)

        torch.ops.sglang.m_grouped_w8a8_gemm_nt_masked(
            q_a2_all, q_a2_scale,
            w2_weight, w2_scales,
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        assert (
            envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ), "W4AFP8 does not support FP8 dispatch; please set SGLANG_DEEPEP_BF16_DISPATCH=1."
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
            hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
                dispatch_output
            )

            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")
                if not input_quant and not isinstance(
                    self.quant_method, NPUW4A16Int4DynamicMoEMethod
                ):
                    hidden_states, hidden_states_scale = torch_npu.npu_dynamic_quant(
                        hidden_states
                    )
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
            (
                hidden_states,
                hidden_states_scale,
                topk_ids,
                topk_weights,
                group_list,
                _,
            ) = dispatch_output

            group_list = group_list.to(torch.int64)

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        else:
            raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")

        return hidden_states


class NpuFuseEPMoE(DeepEPMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )

        self.quant_method.process_weights_after_loading = (
            self._process_weights_after_loading
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
            gmm1_permuted_weight=self.w13_weight,
            gmm1_permuted_weight_scale=self.w13_weight_scale,
            gmm2_weight=self.w2_weight,
            gmm2_weight_scale=self.w2_weight_scale,
        ).hidden_state

    def release_weight_cache(self, weight: torch.Tensor):
        # .contiguous() introduces additional memory overhead and needs to be released using resize_(0)
        origin_weight = weight.data.transpose(1, 2)
        new_weight = origin_weight.contiguous()
        origin_weight.untyped_storage().resize_(0)
        return new_weight

    def permute_w13_weight_scale(self, w: torch.Tensor, tile_n: int):
        if tile_n % 2 != 0:
            raise ValueError(f"tile_n must be even, got {tile_n}")

        *dims, n = w.shape
        if n % tile_n != 0:
            raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

        w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)

        # Permute the last two dimensions.
        perm_order = list(range(len(dims))) + [-2, -3, -1]
        w_permuted = w_reshaped.permute(perm_order)

        return w_permuted.reshape(*dims, n)

    def reshape_w13_weight(self, weight: torch.Tensor, dim: int, chunk_size: int = 64):
        # Achieving greater computing power through reshape on Ascend.
        original_shape = weight.shape
        if dim < 0:
            dim += len(original_shape)

        if original_shape[dim] % (2 * chunk_size) != 0:
            raise ValueError(
                f"Dimension {dim} size {original_shape[dim]} must be divisible by {2 * chunk_size}"
            )

        new_shape = (
            *original_shape[:dim],
            2,
            original_shape[dim] // (2 * chunk_size),
            chunk_size,
            *original_shape[dim + 1 :],
        )

        weight = weight.view(new_shape)
        weight = weight.transpose(dim, dim + 1).contiguous()

        return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])

    def _process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13 = self.release_weight_cache(layer.w13_weight)
        torch_npu.npu_format_cast_(w13, 2)
        cpu_w13 = w13.cpu()
        w13 = self.reshape_w13_weight(cpu_w13, -1).npu()
        torch_npu.npu_format_cast_(w13, 29)
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)

        w2 = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)

        w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
        w13_scale = self.permute_w13_weight_scale(w13_scale, 128)
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale.to(torch.float32), requires_grad=False
        )

        w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale.to(torch.float32), requires_grad=False
        )

        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
        return DeepEPMoE
    if get_moe_a2a_backend().is_ascend_fuseep():
        return NpuFuseEPMoE

    if get_moe_runner_backend().is_flashinfer_trtllm():
        # NEW: Direct FP4 detection (bypasses EP requirements)
        # Check for FP4 quantization with TRTLLM flag, regardless of EP
        # FlashInferFP4MoE must be paired with ModelOptNvFp4FusedMoEMethod.
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFP4MoE

            return FlashInferFP4MoE
        elif (
            quant_config is None
            or quant_config.get_name() == "fp8"
            or quant_config.get_name() == "modelopt_fp8"
            or quant_config.get_name() == "compressed_tensors"
        ):
            # FlashInferFusedMoE support bf16, fp8 and compressed_tensors
            return FlashInferFusedMoE

    if get_moe_runner_backend().is_flashinfer_cutlass():
        return FusedMoE
    return FusedMoE
