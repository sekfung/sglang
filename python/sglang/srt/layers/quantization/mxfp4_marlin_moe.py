from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import log_info_on_rank0, set_weight_attrs
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Mxfp4MarlinMoEMethod:
    """MXFP4/NVFP4 MoE quantization method using the Marlin-compatible path."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        runner_backend = MoeRunnerBackend.MARLIN
        self.runner = MoeRunner(runner_backend, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        layer._dsv4_mxfp4_backend = None  # set in process_weights_after_loading
        # Nvidia's DeepSeek-V4-Flash-NVFP4 checkpoint stores expert FP4 weights
        # with one E4M3 block scale per 16 unpacked K values.
        fp4_block_k = 16

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

        tensor_scale_attrs = dict(extra_weight_attrs)
        tensor_scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.TENSOR.value
        num_shards = 2 if getattr(layer.moe_runner_config, "is_gated", True) else 1

        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
        set_weight_attrs(w13_weight_scale_2, tensor_scale_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
        set_weight_attrs(w2_weight_scale_2, tensor_scale_attrs)

        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, tensor_scale_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, tensor_scale_attrs)

    @staticmethod
    def _expand_w13_global_scale(
        block_scale: torch.Tensor, global_scale: torch.Tensor
    ) -> torch.Tensor:
        gs = global_scale.to(device=block_scale.device, dtype=torch.float32)
        if gs.ndim == 2 and gs.shape[1] == 2:
            rows_per_half = block_scale.shape[1] // 2
            gs = gs.repeat_interleave(rows_per_half, dim=1)
        while gs.ndim < block_scale.ndim:
            gs = gs.unsqueeze(-1)
        return gs

    @classmethod
    def _fold_global_scale_for_triton(
        cls, block_scale: torch.Tensor, global_scale: torch.Tensor, *, is_w13: bool
    ) -> torch.Tensor:
        block_scale_f32 = block_scale.to(torch.float32)
        if is_w13:
            gs = cls._expand_w13_global_scale(block_scale_f32, global_scale)
        else:
            gs = global_scale.to(
                device=block_scale.device, dtype=torch.float32
            ).view(-1, 1, 1)
        return (block_scale_f32 * gs).contiguous()

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        # Let the FP8 base method handle ROCm normalization, etc.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        if not is_sm90_supported() and not is_sm120_supported():
            raise RuntimeError(
                "DeepSeekV4 MXFP4 Marlin fallback requires Hopper/SM90 or above."
            )

        # SM120: Skip Marlin repacking, keep original weight format
        # for Triton dequant kernel (Marlin kernel produces NaN on SM120)
        if is_sm120_supported():
            log_info_on_rank0(
                logger,
                f"SM120 detected: using PyTorch MXFP4 MoE fallback "
                f"(layer: {self.prefix})...",
            )
            # Keep weights in original packed int8 format
            # and precompute dequant scales for the Triton fallback.
            layer._sm120_triton_w13_scale = self._fold_global_scale_for_triton(
                layer.w13_weight_scale.data,
                layer.w13_weight_scale_2.data,
                is_w13=True,
            )
            layer._sm120_triton_w2_scale = self._fold_global_scale_for_triton(
                layer.w2_weight_scale.data,
                layer.w2_weight_scale_2.data,
                is_w13=False,
            )
            layer._dsv4_mxfp4_backend = "sm120_triton"

            # FlashInfer's SM120 b12x W4A4 path currently does not match the
            # NVIDIA DeepSeek-V4-Flash-NVFP4 checkpoint scale semantics used by
            # this wrapper. Keep the correct Triton path even when the old opt-in
            # env is set; otherwise decode can produce corrupted text.
            from sglang.srt.environ import envs

            if envs.SGLANG_OPT_USE_SM120_CUTEDSL_MOE.get():
                logger.warning_once(
                    "SGLANG_OPT_USE_SM120_CUTEDSL_MOE is ignored for "
                    "DeepSeek-V4 NVFP4 experts on SM120 because the b12x W4A4 "
                    "scale mapping is not numerically compatible yet; using "
                    "sm120_triton for correctness."
                )
            return

        if not check_moe_marlin_supports_layer(layer, 32):
            raise RuntimeError(
                "Current DeepSeekV4 MoE layer does not satisfy Marlin constraints."
            )

        # NOTE: the Marlin MoE runner consumes w13 in the checkpoint's
        # native ``[w1; w3]`` order -- see ``silu_and_mul`` in
        # fused_marlin_moe.py which expects ``gate = intermediate[:, :N]``
        # (first half) and ``up = intermediate[:, N:]`` (second half).
        # Unlike the flashinfer trtllm_fp4 kernel (which wants [w3, w1]),
        # we must *not* call ``reorder_w1w3_to_w3w1`` here.

        log_info_on_rank0(
            logger,
            f"Preparing DeepSeekV4 MXFP4 experts for Marlin backend "
            f"(layer: {self.prefix})...",
        )
        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._dsv4_mxfp4_backend = "marlin"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        # SM120: flashinfer native CuTe-DSL fused MoE (opt-in, fastest path)
        if layer._dsv4_mxfp4_backend == "sm120_cutedsl":
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
                Sm120CuteDslMxfp4MoeQuantInfo,
            )

            p = layer._sm120_cutedsl_packed
            quant_info = Sm120CuteDslMxfp4MoeQuantInfo(
                w13_weight=p.w13,
                w13_weight_sf=p.w13_scale,
                w1_alpha=p.w13_global_scale,
                w2_weight=p.w2,
                w2_weight_sf=p.w2_scale,
                w2_alpha=p.w2_global_scale,
                fc2_input_scale=p.fc2_input_scale,
                num_experts=p.w13.shape[0],
                num_local_experts=p.w13.shape[0],
                top_k=topk_output.topk_ids.shape[-1],
                quant_mode=getattr(p, "quant_mode", "nvfp4"),
            )
            return self.runner.run(dispatch_output, quant_info)

        # SM120: use Triton fused dequant+GEMM (Marlin kernel produces NaN on SM120)
        if layer._dsv4_mxfp4_backend == "sm120_triton":
            from sglang.srt.layers.moe.fused_moe_triton.mxfp4_moe_sm120_triton import (
                mxfp4_moe_forward_triton,
            )

            hidden_states = dispatch_output.hidden_states
            w13 = layer.w13_weight.data
            w2 = layer.w2_weight.data
            w13_scale = layer._sm120_triton_w13_scale
            w2_scale = layer._sm120_triton_w2_scale
            intermediate_size = w13.shape[1] // 2
            hidden_size = w13.shape[2] * 2

            output = mxfp4_moe_forward_triton(
                hidden_states=hidden_states,
                w13_packed=w13,
                w2_packed=w2,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                routed_scaling_factor=(
                    self.runner.config.routed_scaling_factor
                    if hasattr(self.runner, "config")
                    else None
                ),
                clamp_limit=(
                    self.runner.config.swiglu_limit
                    if hasattr(self.runner, "config")
                    else None
                ),
            )
            return StandardCombineInput(hidden_states=output)

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            is_k_full=True,
        )
        runner_output = self.runner.run(dispatch_output, quant_info=quant_info)

        return StandardCombineInput(hidden_states=runner_output.hidden_states)
