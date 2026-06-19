"""SM120 native NVFP4 (E4M3 block-scale) fused MoE via flashinfer b12x CuTe-DSL.

Wraps ``flashinfer.b12x_fused_moe`` (sekfung/flashinfer feat/sm120) for the
Nvidia DeepSeek-V4-Flash-NVFP4 checkpoint on RTX PRO 6000 / SM120.  Off by
default; gated by ``SGLANG_OPT_USE_SM120_CUTEDSL_MOE``.

The Nvidia checkpoint stores MoE expert weights as packed FP4 (E2M1) with
16-element E4M3 block scales plus per-expert FP32 ``weight_scale_2``.  At load
time this adapter only swizzles the checkpoint block scales into the MMA layout;
it does not dequantise or requantise the FP4 weights.
"""
import logging
import torch

logger = logging.getLogger(__name__)

_SF_VEC_SIZE = 16  # checkpoint MoE block scale size
_GROUP_SIZE = 16   # E4M3 scale group size (unchanged from nvfp4)


def _fp4_dequant_lut():
    """Pre-compute FP4 E2M1 lookup table (torch.float32)."""
    if not hasattr(_fp4_dequant_lut, "_cache"):
        f16 = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
        )
        _fp4_dequant_lut._cache = f16
    return _fp4_dequant_lut._cache


def _dequant_and_requant_block(w_uint8, scale_f32, sf_vec_size):
    """Dequant packed FP4 with float32 E4M3 scales, re-quant to b12x format.

    Returns ``(w_q, w_sf_swizzled)`` where ``w_q`` is the packed FP4 weight
    tensor (same shape as input) and ``w_sf_swizzled`` is the swizzled 2D
    scale-factor tensor ready for ``convert_sf_to_mma_layout(sf_vec_size)``.
    """
    from flashinfer.fp4_quantization import fp4_quantize

    lut = _fp4_dequant_lut().to(device=w_uint8.device)
    # w_uint8: [M, K//2] — each byte holds two FP4 nibbles
    nibbles_lo = w_uint8 & 0x0F
    nibbles_hi = (w_uint8 >> 4) & 0x0F
    vals_lo = lut[nibbles_lo.long()]
    vals_hi = lut[nibbles_hi.long()]
    # Interleave: [lo0, hi0, lo1, hi1, ...] along the last dim
    w_bf16 = torch.stack([vals_lo, vals_hi], dim=-1).reshape(
        w_uint8.shape[0], w_uint8.shape[1] * 2
    )
    # scale_f32: [M, K//sf_vec_size] — one scale per sf_vec_size elements
    # Expand scale to match each element
    s = scale_f32.repeat_interleave(sf_vec_size, dim=1)
    # Adjust if last dim doesn't divide evenly (shouldn't happen)
    s = s[:, : w_bf16.shape[1]]
    # Dequant: fp4_value * scale = bf16 weight
    w_deq = w_bf16 * s
    w_deq_bf16 = w_deq.to(torch.bfloat16)

    # Re-quant with the same sf_vec_size (no information loss beyond
    # the original FP4 quantisation — the block size is unchanged).
    return fp4_quantize(
        w_deq_bf16,
        global_scale=torch.ones(1, device=w_uint8.device, dtype=torch.float32),
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )


def _swizzle_blockscale_sf32(scale_f32):
    """Swizzle E4M3 block scales to the interleaved 2D
    layout that ``convert_sf_to_mma_layout`` consumes.

    Mirrors ``swizzle_blockscale`` but accepts either raw float8_e4m3fn scales
    from the checkpoint or float32 tensors used by tests.
    """
    import math
    scale_ndim = scale_f32.ndim
    if scale_ndim == 2:
        scale_f32 = scale_f32.unsqueeze(0)
    B, M, K32 = scale_f32.shape  # K32 = cols // _SF_VEC_SIZE
    M_pad = ((M + 127) // 128) * 128
    K_pad = ((K32 + 3) // 4) * 4
    padded = torch.zeros((B, M_pad, K_pad), dtype=scale_f32.dtype, device=scale_f32.device)
    padded[:B, :M, :K32] = scale_f32
    if padded.dtype == torch.float8_e4m3fn:
        f8 = padded
    else:
        # Convert to float8_e4m3fn (the byte format the MMA converter expects).
        f8 = padded.clamp(-448, 448).to(torch.float8_e4m3fn)
    f8 = f8.reshape(B, M_pad // 128, 4, 32, K_pad // 4, 4)
    swz = f8.permute(0, 1, 4, 3, 2, 5).contiguous()
    out = swz.reshape(B, M_pad, K_pad)
    if scale_ndim == 2:
        out = out.squeeze(0)
    return out


def prepare_sm120_cutedsl_weights(layer, *, activation: str = "silu"):
    """Convert NVFP4 checkpoint weights to the b12x MMA layout.

    Two transforms are applied so the b12x SM120 kernel matches SGLang's
    SwiGLU semantics and the checkpoint's NVFP4 scale convention:

    1. **Swap the w13 halves.** SGLang loads w13 as ``[gate, up]`` and computes
       ``silu(gate) * up`` (first half gated). The b12x kernel instead computes
       ``silu(second_half) * first_half`` (see flashinfer's
       ``compute_reference_moe_fp4``), so the halves are reordered to
       ``[up, gate]`` together with their block scales.

    2. **Fold ``weight_scale_2`` into the per-block E4M3 scales** (alpha → 1).
       The kernel reuses ``w1_alpha`` as ``input_gs`` — the per-tensor global
       scale used to quantise the *activations* — so passing the weight global
       scale (``weight_scale_2`` ≈ ``2**-13``) as alpha corrupts activation
       quantisation and blows the output up by ~5 orders of magnitude. Folding
       keeps alpha = 1 (correct ``input_gs``) and is lossless here because the
       checkpoint block scales are powers of two in ``[32, 256]``, so
       ``block_scale * 2**-13`` stays exactly representable in E4M3.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    w13 = layer.w13_weight.data              # [E, 2*n, k/2]   uint8  [gate, up]
    w2  = layer.w2_weight.data               # [E, k,   n/2]   uint8
    w13_s = layer.w13_weight_scale.data      # [E, 2*n, k/16]  float8_e4m3fn
    w2_s  = layer.w2_weight_scale.data       # [E, k,   n/16]  float8_e4m3fn
    w13_global_scale = layer.w13_weight_scale_2.data  # [E, 2] or [E]  float32
    w2_global_scale = layer.w2_weight_scale_2.data    # [E]           float32

    num_experts = w13.shape[0]
    device = w13.device
    w13_rows = w13.shape[1]            # 2 * intermediate_per_partition
    hidden_size = w13.shape[2] * 2     # unpacked K
    w2_rows = w2.shape[1]              # hidden_size
    intermediate_size = w2.shape[2] * 2  # unpacked N
    n_half = w13_rows // 2            # intermediate_per_partition

    # 1. Swap w13 halves [gate, up] -> [up, gate] (weights + block scales).
    w13 = torch.cat([w13[:, n_half:], w13[:, :n_half]], dim=1).contiguous()
    w13_s = torch.cat([w13_s[:, n_half:], w13_s[:, :n_half]], dim=1)

    # 2. Fold weight_scale_2 into block scales (per-half for w13). After the
    #    swap, the first half is "up" and the second half is "gate", so the
    #    per-half global scales are reordered to match.
    gs = w13_global_scale.to(torch.float32)
    if gs.ndim == 2 and gs.shape[1] >= 2:
        gs_swapped = gs[:, [1, 0]]                    # [up_gs, gate_gs]
    else:
        gs_swapped = gs.reshape(num_experts, 1).expand(num_experts, 2)
    gs_rows = gs_swapped.repeat_interleave(n_half, dim=1)  # [E, 2*n]
    w13_sf_folded = (w13_s.float() * gs_rows[..., None]).to(torch.float8_e4m3fn)
    w2_sf_folded = (
        w2_s.float() * w2_global_scale.to(torch.float32).view(num_experts, 1, 1)
    ).to(torch.float8_e4m3fn)

    # 3. Swizzle (per-expert, 3D) then convert to the 6D MMA layout.
    w13_sf_mma = convert_sf_to_mma_layout(
        _swizzle_blockscale_sf32(w13_sf_folded).contiguous().view(torch.uint8).reshape(-1),
        m=w13_rows, k=hidden_size, num_groups=num_experts, sf_vec_size=_SF_VEC_SIZE,
    )
    w2_sf_mma = convert_sf_to_mma_layout(
        _swizzle_blockscale_sf32(w2_sf_folded).contiguous().view(torch.uint8).reshape(-1),
        m=w2_rows, k=intermediate_size, num_groups=num_experts, sf_vec_size=_SF_VEC_SIZE,
    )

    # 4. Global scales are folded in; alpha = 1. fc2_input_scale = 1 (the kernel
    #    does dynamic per-block FC2-input requant).
    ones_e = torch.ones(num_experts, dtype=torch.float32, device=device)

    layer._sm120_cutedsl_packed = type("_Packed", (), {
        "w13": w13,
        "w13_scale": w13_sf_mma,
        "w13_global_scale": ones_e,
        "w2": w2,
        "w2_scale": w2_sf_mma,
        "w2_global_scale": ones_e,
        "fc2_input_scale": torch.ones(1, dtype=torch.float32, device=device),
        "quant_mode": "nvfp4",
        "workspace": None,
    })()
    layer._dsv4_mxfp4_backend = "sm120_cutedsl"

    # Release tensors the b12x path no longer needs so we don't keep a second
    # full-size copy of w13 (the swap) plus the original block scales alive for
    # every layer. The cutedsl forward reads only ``_sm120_cutedsl_packed``.
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    layer.w13_weight.data = w13                  # swapped weight replaces [gate,up]
    layer.w13_weight_scale.data = empty          # consumed into the MMA scale
    layer.w2_weight_scale.data = empty
    for attr in ("_sm120_triton_w13_scale", "_sm120_triton_w2_scale"):
        if getattr(layer, attr, None) is not None:
            setattr(layer, attr, None)


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
    """Run the SM120 CuTe-DSL NVFP4-sf32 fused MoE for one layer."""
    from flashinfer import b12x_fused_moe

    p = layer._sm120_cutedsl_packed
    num_tokens, hidden_size = hidden_states.shape
    scatter_output = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device,
    )
    return b12x_fused_moe(
        x=hidden_states,
        w1_weight=p.w13,
        w1_weight_sf=p.w13_scale,
        w2_weight=p.w2,
        w2_weight_sf=p.w2_scale,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        num_experts=num_experts,
        top_k=top_k,
        w1_alpha=p.w13_global_scale,
        w2_alpha=p.w2_global_scale,
        fc2_input_scale=p.fc2_input_scale,
        num_local_experts=p.w13.shape[0],
        activation=activation,
        quant_mode="nvfp4_sf32",
        output=scatter_output,
    )
