"""SM120 native NVFP4 fused MoE via flashinfer b12x CuTe-DSL.

Wraps ``flashinfer.b12x_fused_moe`` (the same kernel used by the MXFP4
SM120 adapter) for NVFP4-prequantized checkpoints (e.g. Nvidia's
DeepSeek-V4-Flash-NVFP4) on RTX PRO 6000 / SM120.  Off by default; gated
by ``SGLANG_OPT_USE_SM120_CUTEDSL_MOE`` (shared with the MXFP4 adapter).

The kernel fuses token dispatch, W1 GEMM, SwiGLU/ReLU2, and W2 GEMM into
one call.  Following vLLM #40082, per-tensor ``weight_scale_2`` is folded
into the per-block E4M3 scales so that ``w1_alpha``/``w2_alpha`` become
1.0 and ``fc2_input_scale`` is set to ones (the kernel does dynamic
per-block FC2-input requantisation internally).

Differences from the MXFP4 (W4A16) adapter:
  - Block scales are 16-element E4M3 (NVFP4) not 32-element E8M0 (MXFP4).
  - The checkpoint supplies *both* per-block and per-tensor global scales.
  - ``fc2_input_scale`` is passed (as ones) because the NVFP4 dispatch
    path reads it.
"""
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_SF_VEC_SIZE = 16  # NVFP4: 16-element E4M3 block scales


def _fold_global_into_block_scales(
    block_scale: torch.Tensor,  # [E, rows, cols/16] float8_e4m3fn
    global_scale: torch.Tensor,  # [E], [E, 1], or [E, 2] float32
) -> torch.Tensor:
    """Fold per-expert global scale into per-block scales → alpha = 1.0.

    Following vLLM #40082 but additionally handling SGLang's ``[E, 2]``
    shape for gated w13 (separate gate/up global scales): the 2 halves are
    repeat-interleaved to match the n rows of each half.
    """
    gs = global_scale.to(torch.float32)
    # If gs is [E, 2] (gated w13), expand to [E, 2*n, 1] by repeating
    # each half's scale across its n intermediate rows.
    if gs.ndim == 2 and gs.shape[1] == 2:
        # block_scale has rows = 2 * intermediate_size
        n_per_half = block_scale.shape[1] // 2
        gs = gs.repeat_interleave(n_per_half, dim=1)
    while gs.ndim < block_scale.ndim:
        gs = gs.unsqueeze(-1)
    folded = block_scale.float() * gs
    return folded.to(torch.float8_e4m3fn)


def _swizzle_blockscale_local(scale: torch.Tensor) -> torch.Tensor:
    """Local copy of sglang's ``swizzle_blockscale`` (pure PyTorch — avoids the
    full sglang import chain at module level, allowing standalone verification)."""
    assert scale.dtype == torch.float8_e4m3fn
    scale_ndim = scale.ndim
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
    assert scale.ndim == 3
    B, M, K = scale.shape
    round_up_multiple = lambda x, m: (x + m - 1) // m * m
    M_padded = round_up_multiple(M, 128)
    K_padded = round_up_multiple(K, 4)
    padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)
    padded_scale[:B, :M, :K] = scale
    batches, rows, cols = padded_scale.shape
    assert rows % 128 == 0
    assert cols % 4 == 0
    padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
    swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
    swizzled_scale = swizzled_scale.contiguous().cuda()
    return (
        swizzled_scale.reshape(M_padded, K_padded)
        if scale_ndim == 2
        else swizzled_scale.reshape(B, M_padded, K_padded)
    )


def _block_scale_to_mma_layout(
    block_scale_f8: torch.Tensor,   # [E, rows, cols/16] float8_e4m3fn (FOLDED)
    num_experts: int,
    rows: int,
    cols: int,                      # unpacked K (hidden_size or intermediate_size)
) -> torch.Tensor:
    """Swizzle folded block scales then convert to 6D MMA layout for b12x.

    Returns a strided 6D view (NOT contiguous) — exactly what the kernel's
    ``convert_sf_from_mma_layout`` expects.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    swizzled_flat = (
        _swizzle_blockscale_local(block_scale_f8)
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
    )
    return convert_sf_to_mma_layout(
        swizzled_flat,
        m=rows,
        k=cols,
        num_groups=num_experts,
        sf_vec_size=_SF_VEC_SIZE,
    )


def prepare_sm120_cutedsl_nvfp4_weights(
    layer: "torch.nn.Module",
    *,
    activation: str = "silu",
) -> None:
    """Pack NVFP4 expert weights for the b12x fused-MoE kernel.

    Called once from ``ModelOptNvFp4FusedMoEMethod.process_weights_after_loading``
    when ``SGLANG_OPT_USE_SM120_CUTEDSL_MOE`` is enabled.  Stores the packed
    tensors on ``layer._sm120_cutedsl_nvfp4``.
    """
    w13_weight = layer.w13_weight.data  # [E, 2*n, k/2] uint8 FP4x2
    w2_weight = layer.w2_weight.data    # [E, k, n/2] uint8 FP4x2
    w13_bs = layer.w13_weight_scale.data  # [E, 2*n, k/16] float8_e4m3fn
    w2_bs = layer.w2_weight_scale.data    # [E, k, n/16] float8_e4m3fn
    w13_gs = layer.w13_weight_scale_2.data  # [E, 2] or [E] float32
    w2_gs = layer.w2_weight_scale_2.data    # [E] float32

    device = w13_weight.device
    num_experts = w13_weight.shape[0]
    is_gated = activation == "silu"
    w13_rows = w13_weight.shape[1]          # 2*n (gated) or n (relu2)
    hidden_size = w13_weight.shape[2] * 2    # unpacked K
    intermediate_size = w2_weight.shape[2] * 2  # unpacked N

    # 1. Fold global per-tensor scale into block scales → alpha = 1.0
    w13_sf_folded = _fold_global_into_block_scales(w13_bs, w13_gs)
    w2_sf_folded = _fold_global_into_block_scales(w2_bs, w2_gs)

    # 2. Swizzle + convert to MMA 6D layout
    w13_sf_mma = _block_scale_to_mma_layout(
        w13_sf_folded, num_experts, w13_rows, hidden_size
    )
    w2_sf_mma = _block_scale_to_mma_layout(
        w2_sf_folded, num_experts, hidden_size, intermediate_size
    )

    # 3. Alphas = ones (scales are self-contained in the blocks)
    ones_e = torch.ones(num_experts, dtype=torch.float32, device=device)

    # 4. FC2 input scale = ones (b12x NVFP4 epilogue does dynamic per-block
    #    FC2-input requant; the global scale is neutral).  Must be a valid
    #    tensor because the NVFP4 dispatch path reads it.
    fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)

    layer._sm120_cutedsl_nvfp4 = {
        "w13_weight": w13_weight,
        "w13_sf": w13_sf_mma,
        "w13_alpha": ones_e,
        "w2_weight": w2_weight,
        "w2_sf": w2_sf_mma,
        "w2_alpha": ones_e,
        "fc2_input_scale": fc2_input_scale,
        "is_gated": is_gated,
    }
    layer._nvfp4_backend = "sm120_cutedsl"


def nvfp4_moe_forward_sm120_cutedsl(
    layer: "torch.nn.Module",
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    activation: str = "silu",
    scatter_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the SM120 CuTe-DSL NVFP4 fused MoE for one layer."""
    from flashinfer import b12x_fused_moe as b12x_fused_moe

    p = layer._sm120_cutedsl_nvfp4
    num_tokens, hidden_size = hidden_states.shape

    if scatter_output is None:
        scatter_output = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    return b12x_fused_moe(
        x=hidden_states,
        w1_weight=p["w13_weight"],
        w1_weight_sf=p["w13_sf"],
        w2_weight=p["w2_weight"],
        w2_weight_sf=p["w2_sf"],
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        num_experts=num_experts,
        top_k=top_k,
        w1_alpha=p["w13_alpha"],
        w2_alpha=p["w2_alpha"],
        fc2_input_scale=p["fc2_input_scale"],
        num_local_experts=p["w13_weight"].shape[0],
        activation=activation,
        quant_mode="nvfp4",
        output=scatter_output,
    )
