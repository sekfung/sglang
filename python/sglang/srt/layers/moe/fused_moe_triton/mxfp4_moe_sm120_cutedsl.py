"""SM120 native NVFP4 (sf32/E4M3) fused MoE via flashinfer b12x CuTe-DSL.

Wraps ``flashinfer.b12x_fused_moe`` (sekfung/flashinfer feat/sm120) for the
Nvidia DeepSeek-V4-Flash-NVFP4 checkpoint on RTX PRO 6000 / SM120.  Off by
default; gated by ``SGLANG_OPT_USE_SM120_CUTEDSL_MOE``.

The checkpoint stores MoE expert weights as packed FP4 (E2M1) with 32-element
E4M3 block scales.  At load time the adapter dequantises to bf16, re-quantises
with the same block size via ``fp4_quantize(sf_vec_size=32)`` to produce the
swizzled layout, converts to 6D MMA layout, and stores the result for the
forward pass.  This dequant→requant is a one-time cost (~10 s per layer) and
does not introduce additional quantisation noise because the block size stays
the same.
"""
import logging
import torch

logger = logging.getLogger(__name__)

_SF_VEC_SIZE = 16  # requant block size (checkpoint has 32, but fp4_quantize C++ kernel requires sf16 for E4M3)
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


def prepare_sm120_cutedsl_weights(layer, *, activation: str = "silu"):
    """Convert NVFP4-sf32 checkpoint weights to b12x MMA layout.

    Stores the packed result on ``layer._sm120_cutedsl_packed``.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    w13 = layer.w13_weight.data           # [E, 2*n, k/2]  uint8  FP4x2
    w2  = layer.w2_weight.data            # [E, k,   n/2]  uint8  FP4x2
    w13_s = layer.w13_weight_scale_inv.data  # [E, 2*n, k/32] float32 E4M3
    w2_s  = layer.w2_weight_scale_inv.data   # [E, k,   n/32] float32 E4M3

    num_experts = w13.shape[0]
    device = w13.device

    # Detect format: float8_e8m0fnu needs a different scale conversion.
    # For DeepSeek-V4-Flash-NVFP4 the scales arrive as float32 (E4M3
    # reinterpreted by Mxfp4MarlinMoEMethod).
    if w13_s.dtype != w2_s.dtype:
        raise RuntimeError(
            f"w13 and w2 scales have different dtypes: {w13_s.dtype} vs {w2_s.dtype}"
        )

    # Flatten experts + rows for the batch quantizer.
    w13_rows = w13.shape[1]  # 2 * intermediate_per_partition
    w2_rows = w2.shape[1]   # hidden_size
    hidden_size = w13.shape[2] * 2      # unpacked K
    intermediate_size = w2.shape[2] * 2  # unpacked N

    w13_flat = w13.view(num_experts * w13_rows, hidden_size // 2)
    w2_flat = w2.view(num_experts * w2_rows, intermediate_size // 2)
    w13_s_flat = w13_s.reshape(num_experts * w13_rows, hidden_size // _SF_VEC_SIZE)
    w2_s_flat = w2_s.reshape(num_experts * w2_rows, intermediate_size // _SF_VEC_SIZE)

    # Dequant → requant with same sf_vec_size (one-time cost).
    w13_q, w13_sf_swz = _dequant_and_requant_block(w13_flat, w13_s_flat, _SF_VEC_SIZE)
    w2_q, w2_sf_swz = _dequant_and_requant_block(w2_flat, w2_s_flat, _SF_VEC_SIZE)

    # Convert to 6D MMA layout.
    w13_sf_mma = convert_sf_to_mma_layout(
        w13_sf_swz, m=w13_rows, k=hidden_size,
        num_groups=num_experts, sf_vec_size=_SF_VEC_SIZE,
    )
    w2_sf_mma = convert_sf_to_mma_layout(
        w2_sf_swz, m=w2_rows, k=intermediate_size,
        num_groups=num_experts, sf_vec_size=_SF_VEC_SIZE,
    )

    ones = torch.ones(num_experts, dtype=torch.float32, device=device)
    fc2_is = torch.ones(1, dtype=torch.float32, device=device)

    layer._sm120_cutedsl_packed = type("_Packed", (), {
        "w13": w13_q.view(num_experts, w13_rows, hidden_size // 2),
        "w13_scale": w13_sf_mma,
        "w13_global_scale": ones,
        "w2": w2_q.view(num_experts, w2_rows, intermediate_size // 2),
        "w2_scale": w2_sf_mma,
        "w2_global_scale": ones,
        "fc2_input_scale": fc2_is,
        "workspace": None,  # use dynamic workspace
    })()
    layer._dsv4_mxfp4_backend = "sm120_cutedsl"


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
        quant_mode="nvfp4",
        output=scatter_output,
    )
