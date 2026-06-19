"""SM120 native MXFP4 (W4A16) fused MoE via flashinfer CuTe-DSL.

Wraps ``flashinfer.fused_moe.cute_dsl.blackwell_sm12x`` (sekfung/flashinfer
feat/sm120). Replaces the ``sm120_triton`` MXFP4 MoE fallback for DeepSeek V4
on RTX PRO 6000 / SM120. Off by default; gated by
``SGLANG_OPT_USE_SM120_CUTEDSL_MOE``.

The kernel fuses token dispatch, W1 GEMM, SwiGLU and W2 GEMM into one call
(``launch_sm120_moe``). Weights are pre-packed once at load time via
``prepare_w4a16_packed_weights`` and cached on the layer.

The SGLang-loaded DeepSeek V4 MXFP4 expert weights line up with the kernel's
expected layout directly:
  w13_weight        (E, 2*intermediate, hidden//2) uint8  -> w13_fp4
  w2_weight         (E, hidden,         inter//2)   uint8  -> w2_fp4
  w13_weight_scale_inv / w2_weight_scale_inv (e8m0/fp32)   -> blockscale
MXFP4 has no per-tensor global scale, so global_scale defaults to ones (fp32).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def prepare_sm120_cutedsl_weights(layer, *, activation: str = "silu"):
    """Pack DeepSeek V4 MXFP4 expert weights for the SM120 CuTe-DSL kernel.

    Stores the packed result on ``layer._sm120_cutedsl_packed`` and a reusable
    workspace handle. Called once from ``process_weights_after_loading``.
    """
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        prepare_w4a16_packed_weights,
    )

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale_inv.data
    w2_scale = layer.w2_weight_scale_inv.data

    num_experts = w13.shape[0]
    device = w13.device
    # MXFP4: e8m0 block scales only, no per-tensor global scale -> ones.
    w13_gscale = torch.ones(num_experts, dtype=torch.float32, device=device)
    w2_gscale = torch.ones(num_experts, dtype=torch.float32, device=device)

    packed = prepare_w4a16_packed_weights(
        w13_fp4=w13.view(torch.uint8),
        w13_blockscale=w13_scale,
        w13_global_scale=w13_gscale,
        w2_fp4=w2.view(torch.uint8),
        w2_blockscale=w2_scale,
        w2_global_scale=w2_gscale,
        activation=activation,
        params_dtype=torch.bfloat16,
        source_format="modelopt",
    )
    layer._sm120_cutedsl_packed = packed
    logger.info_once(
        "SM120 CuTe-DSL MXFP4 MoE enabled (experts=%d)", num_experts
    )


def mxfp4_moe_forward_sm120_cutedsl(
    layer,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    activation: str = "silu",
) -> torch.Tensor:
    """Run the SM120 CuTe-DSL W4A16 fused MoE for one layer."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import launch_sm120_moe

    packed = layer._sm120_cutedsl_packed
    num_tokens, hidden_size = hidden_states.shape
    scatter_output = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )

    launch_sm120_moe(
        a=hidden_states,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights,
        w1_weight=packed.w13,
        w1_weight_sf=packed.w13_scale,
        w1_alpha=packed.w13_global_scale,
        w2_weight=packed.w2,
        w2_weight_sf=packed.w2_scale,
        w2_alpha=packed.w2_global_scale,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_experts,
        scatter_output=scatter_output,
        activation=activation,
        # MXFP4 weights are served as W4A16 (4-bit weights, bf16 activations):
        # this matches prepare_w4a16_packed_weights above and routes to
        # _launch_sm120_w4a16_moe. Without an explicit quant_mode the dispatch
        # normalizes activation_precision="fp4" to "nvfp4", which both mismatches
        # the W4A16-packed weights and trips the "fc2_input_scale is required"
        # guard, so the path would crash the moment the env flag is enabled.
        quant_mode="w4a16",
        source_format="modelopt",
        _workspace=packed.workspace,
    )
    return scatter_output
