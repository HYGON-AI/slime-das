from typing import Any, Callable, Dict, List, Optional
# from sglang.srt.layers.moe.token_dispatcher.base import CombineInput

import torch
from sglang.srt.utils import set_weight_attrs, get_bool_env_var
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from torch.nn.parameter import Parameter
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.quantization.w4a8_utils import w4a8_weight_repack_impl
from sglang.srt.layers.quantization.base_config import (FusedMoEMethodBase, QuantizeMethodBase)
from sglang.srt.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig

try:
    from lmslim.layers.fused_moe.fuse_moe_w4a8_marlin import fused_experts_impl_w4a8_marlin
except Exception:
    print("INFO: Please install lmslim if you want to infer the quantitative model of moe.\n")


class MarlinMoeWorkspace:
    """
    Singleton manager for device-specific workspace buffers used by w4a8 Marlin-MoE.
    global_reduce_buffer will take 1.5MB * cus (about 120MB for BW200) memoery in each device
    """
    _instances = {}
    def __new__(cls, device):
        if device not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[device] = instance
        return cls._instances[device]

    def __init__(self, device):
        if self._initialized:
            return
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.zeros(
            500, dtype=torch.int, device=device, requires_grad=False
        )
        self.global_reduce_buffer = torch.zeros(
            sms * 6 * 128 * 512, dtype=torch.int, device=device, requires_grad=False
        )
        self._initialized = True

    def get_buffers(self):
        return self.workspace, self.global_reduce_buffer

def baseline_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    scales= scale_a* scale_b.T
    gemmout= torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32))
    output = (scales *gemmout).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output.to(out_dtype)


class SlimQuantW4A8Int8MarlinConfig(QuantizationConfig):
    """Config class for W4A8 Int8 Quantization.
    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self):
        pass

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "slimquant_w4a8_marlin"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlimQuantW4A8Int8MarlinConfig":
        return cls()
    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if hf_quant_cfg.get("quant_method") == "slimquant_w4a8" \
                and user_quant in ("slimquant_w4a8_marlin", "slimquant_marlin"):
            return cls.get_name()
        return None
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import (FusedMoE, FusedMoeWeightScaleSupported)

        if isinstance(layer, LinearBase):
            return SlimQuantW4A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return SlimQuantW4A8Int8MarlinMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class SlimQuantW4A8Int8MarlinMoEMethod:
    """MoE method for W4A8INT8 Marlin.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import (FusedMoE, FusedMoeWeightScaleSupported)

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (FusedMoE, FusedMoeWeightScaleSupported)
        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = intermediate_size_per_partition
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size//2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size//2, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

        layer.w13_weight = Parameter(w4a8_weight_repack_impl(layer.w13_weight), requires_grad=False)
        layer.w2_weight = Parameter(w4a8_weight_repack_impl(layer.w2_weight), requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    @torch._dynamo.disable()  # TODO: 性能优化需lmslim/lightop配合
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
       # local_expert_mapping,
    ) :
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        output = fused_experts_impl_w4a8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=True,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=layer.moe_runner_config.activation,
           # expert_map=local_expert_mapping,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
        )
        return StandardCombineInput(hidden_states=output)

    @torch._dynamo.disable()  # TODO: 性能优化需lmslim/lightop配合
    def apply_with_shared_output(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        activation: str = "silu",
        shared_output: Optional[torch.Tensor] = None,
        topk_output=None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        return fused_experts_impl_w4a8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=True,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            shared_output=shared_output,
            i_q=i_q,
            i_s=i_s,
        )

    # def _apply(
    #     self,
    #     layer: torch.nn.Module,
    #     x: torch.Tensor,
    #     router_logits: torch.Tensor,
    #     top_k: int,
    #     #renormalize: bool,
    #     #use_grouped_topk: bool = False,
    #     topk_group: Optional[int] = None,
    #     num_expert_group: Optional[int] = None,
    #     global_num_experts: int = -1,
    #     expert_map: Optional[torch.Tensor] = None,
    #     custom_routing_function: Optional[Callable] = None,
    #     scoring_func: str = "softmax",
    #     e_score_correction_bias: Optional[torch.Tensor] = None,
    #     apply_router_weight_on_input: bool = False,
    #     activation: str = "silu",
    #     enable_eplb: bool = False,
    #     use_nn_moe: Optional[bool] = False,
    #     routed_scaling_factor: Optional[float] = None,
    #     use_fused_gate: Optional[bool] = False,
    #     **_
    # ) -> torch.Tensor:
    #     from sglang.srt.layers.moe.fused_moe_triton import (FusedMoE, FusedMoeWeightScaleSupported)
    #     from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    #     if enable_eplb:
    #         raise NotImplementedError(
    #             "EPLB not supported for `SlimQuantW4A8Int8MarlinMoEMethod` yet.")
    #     # Expert selection
    #     topk_weights, topk_ids = FusedMoE.select_experts(
    #         hidden_states=x,
    #         router_logits=router_logits,
    #         #use_grouped_topk=use_grouped_topk,
    #         top_k=top_k,
    #         #renormalize=renormalize,
    #         topk_group=topk_group,
    #         num_expert_group=num_expert_group,
    #         custom_routing_function=custom_routing_function,
    #         scoring_func=scoring_func,
    #         e_score_correction_bias=e_score_correction_bias,
    #         routed_scaling_factor=routed_scaling_factor,
    #         use_fused_gate=use_fused_gate
    #     )
    #     workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
    #     return fused_experts_impl_w4a8_marlin(
    #         x,
    #         layer.w13_weight,
    #         layer.w2_weight,
    #         topk_weights=topk_weights,
    #         topk_ids=topk_ids,
    #         workspace=workspace,
    #         global_reduce_buffer=global_reduce_buffer,
    #         inplace=True,
    #         use_int4_w4a8=True,
    #         per_channel_quant=True,
    #         activation=activation,
    #         expert_map=expert_map,
    #         apply_router_weight_on_input=apply_router_weight_on_input,
    #         global_num_experts=global_num_experts,
    #         w1_scale=(layer.w13_weight_scale),
    #         w2_scale=(layer.w2_weight_scale),
    #         a1_scale=layer.w13_input_scale,
    #         a2_scale=layer.w2_input_scale,
    #         use_nn_moe=use_nn_moe,
    #     )

    def apply_ep(self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        use_nn_moe: Optional[bool] = False,
        num_local_tokens: Optional[torch.Tensor] = None,
        #config_select_bs: Optional[int] = None,
        routed_scaling_factor: Optional[float] = 1.0,
        shared_output: Optional[torch.Tensor] = None,
        #scales: Optional[torch.Tensor] = None,
        num_recv_tokens_per_expert: List = None,
        **_  ):
            workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
            return fused_experts_impl_w4a8_marlin(
                x,
                w1,
                w2,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                workspace=workspace,
                global_reduce_buffer=global_reduce_buffer,
                inplace=True,
                use_int4_w4a8=True,
                per_channel_quant=True,
                activation=activation,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                use_nn_moe=use_nn_moe,
                shared_output=shared_output,
                routed_scaling_factor=routed_scaling_factor,
                # num_local_tokens=num_local_tokens,
                #config_select_bs=config_select_bs,
                #q_scales=scales

            )
