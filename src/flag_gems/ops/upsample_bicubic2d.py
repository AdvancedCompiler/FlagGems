import math
from typing import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def cubic_weight(d: float, a: tl.constexpr):
    ad = tl.abs(d)
    ad2 = ad * ad
    ad3 = ad2 * ad
    w1 = (a + 2.0) * ad3 - (a + 3.0) * ad2 + 1.0
    w2 = a * ad3 - 5.0 * a * ad2 + 8.0 * a * ad - 4.0 * a
    return tl.where(ad <= 1.0, w1, tl.where(ad < 2.0, w2, tl.zeros_like(ad)))


@triton.jit
def _upsample_bicubic2d(
    in_ptr,
    out_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    strideN,
    strideC,
    strideH,
    strideW,
    out_strideN,
    out_strideC,
    out_strideH,
    out_strideW,
    scale_h,
    scale_w,
    align_corners: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc_h = tl.program_id(axis=0)
    pid_wblk = tl.program_id(axis=1)

    out_y = pid_nc_h % H_out
    nc = pid_nc_h // H_out
    n = nc // C
    c = nc % C

    off_x = pid_wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_x = off_x < W_out

    base_in = n * strideN + c * strideC
    base_out = n * out_strideN + c * out_strideC + out_y * out_strideH

    a = tl.full([1], -0.75, tl.float32)

    fy = tl.full([1], 0.0, tl.float32) + out_y.to(tl.float32)
    in_y = tl.where(
        tl.full([1], align_corners, tl.int32) != 0,
        fy * scale_h,
        (fy + 0.5) * scale_h - 0.5,
    )

    y0f = tl.floor(in_y)
    y0 = y0f.to(tl.int32)
    ty = in_y - y0f

    y_idx_m1 = tl.maximum(0, tl.minimum(H_in - 1, y0 - 1))
    y_idx_0 = tl.maximum(0, tl.minimum(H_in - 1, y0 + 0))
    y_idx_p1 = tl.maximum(0, tl.minimum(H_in - 1, y0 + 1))
    y_idx_p2 = tl.maximum(0, tl.minimum(H_in - 1, y0 + 2))

    dy0 = 1.0 + ty
    dy1 = ty
    dy2 = 1.0 - ty
    dy3 = 2.0 - ty

    wy0 = cubic_weight(dy0, a)
    wy1 = cubic_weight(dy1, a)
    wy2 = cubic_weight(dy2, a)
    wy3 = cubic_weight(dy3, a)

    fx = off_x.to(tl.float32)
    in_x = tl.where(
        tl.full([BLOCK_W], align_corners, tl.int32) != 0,
        fx * scale_w,
        (fx + 0.5) * scale_w - 0.5,
    )

    x0f = tl.floor(in_x)
    x0 = x0f.to(tl.int32)
    tx = in_x - x0f

    x_idx_m1 = tl.maximum(0, tl.minimum(W_in - 1, x0 - 1))
    x_idx_0 = tl.maximum(0, tl.minimum(W_in - 1, x0 + 0))
    x_idx_p1 = tl.maximum(0, tl.minimum(W_in - 1, x0 + 1))
    x_idx_p2 = tl.maximum(0, tl.minimum(W_in - 1, x0 + 2))

    dx0 = 1.0 + tx
    dx1 = tx
    dx2 = 1.0 - tx
    dx3 = 2.0 - tx

    wx0 = cubic_weight(dx0, a)
    wx1 = cubic_weight(dx1, a)
    wx2 = cubic_weight(dx2, a)
    wx3 = cubic_weight(dx3, a)

    ptr_row = in_ptr + (base_in + y_idx_m1 * strideH)
    v00 = tl.load(ptr_row + x_idx_m1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v01 = tl.load(ptr_row + x_idx_0 * strideW, mask=mask_x, other=0).to(tl.float32)
    v02 = tl.load(ptr_row + x_idx_p1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v03 = tl.load(ptr_row + x_idx_p2 * strideW, mask=mask_x, other=0).to(tl.float32)
    row0 = v00 * wx0 + v01 * wx1 + v02 * wx2 + v03 * wx3

    ptr_row = in_ptr + (base_in + y_idx_0 * strideH)
    v10 = tl.load(ptr_row + x_idx_m1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v11 = tl.load(ptr_row + x_idx_0 * strideW, mask=mask_x, other=0).to(tl.float32)
    v12 = tl.load(ptr_row + x_idx_p1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v13 = tl.load(ptr_row + x_idx_p2 * strideW, mask=mask_x, other=0).to(tl.float32)
    row1 = v10 * wx0 + v11 * wx1 + v12 * wx2 + v13 * wx3

    ptr_row = in_ptr + (base_in + y_idx_p1 * strideH)
    v20 = tl.load(ptr_row + x_idx_m1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v21 = tl.load(ptr_row + x_idx_0 * strideW, mask=mask_x, other=0).to(tl.float32)
    v22 = tl.load(ptr_row + x_idx_p1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v23 = tl.load(ptr_row + x_idx_p2 * strideW, mask=mask_x, other=0).to(tl.float32)
    row2 = v20 * wx0 + v21 * wx1 + v22 * wx2 + v23 * wx3

    ptr_row = in_ptr + (base_in + y_idx_p2 * strideH)
    v30 = tl.load(ptr_row + x_idx_m1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v31 = tl.load(ptr_row + x_idx_0 * strideW, mask=mask_x, other=0).to(tl.float32)
    v32 = tl.load(ptr_row + x_idx_p1 * strideW, mask=mask_x, other=0).to(tl.float32)
    v33 = tl.load(ptr_row + x_idx_p2 * strideW, mask=mask_x, other=0).to(tl.float32)
    row3 = v30 * wx0 + v31 * wx1 + v32 * wx2 + v33 * wx3

    out_vals = row0 * wy0 + row1 * wy1 + row2 * wy2 + row3 * wy3

    tl.store(out_ptr + base_out + off_x * out_strideW, out_vals, mask=mask_x)


def upsample_bicubic2d(
    input: torch.Tensor,
    output_size: Sequence[int] | None = None,
    align_corners: bool = False,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> torch.Tensor:
    scale_factors = (scales_h, scales_w)

    if input.dim() != 4:
        raise ValueError("input must be a 4D tensor (N, C, H, W)")
    if output_size is None and scale_factors is None:
        raise ValueError("Either output_size or scale_factors must be provided")

    N, C, H_in, W_in = input.shape

    if output_size is not None:
        if len(output_size) != 2:
            raise ValueError(
                "output_size must be a sequence of two ints (H_out, W_out)"
            )
        H_out, W_out = int(output_size[0]), int(output_size[1])
    else:
        if len(scale_factors) == 2:
            sh, sw = float(scale_factors[0]), float(scale_factors[1])
        elif len(scale_factors) == 1:
            sh = sw = float(scale_factors[0])
        else:
            raise ValueError("scale_factors must have length 1 or 2 for 2D upsampling")
        H_out = max(int(math.floor(H_in * sh)), 1)
        W_out = max(int(math.floor(W_in * sw)), 1)

    if H_out <= 0 or W_out <= 0:
        raise ValueError("Output size must be positive")

    device = input.device
    if not input.is_cuda:
        raise ValueError("This Triton kernel requires CUDA tensors")

    if align_corners:
        scale_h = 0.0 if H_out <= 1 else (H_in - 1.0) / (H_out - 1.0)
        scale_w = 0.0 if W_out <= 1 else (W_in - 1.0) / (W_out - 1.0)
    else:
        scale_h = float(H_in) / float(H_out)
        scale_w = float(W_in) / float(W_out)

    out_fp32 = torch.empty((N, C, H_out, W_out), dtype=torch.float32, device=device)

    sN, sC, sH, sW = input.stride()
    oN, oC, oH, oW = out_fp32.stride()

    BLOCK_W = 128
    grid = (
        N * C * H_out,
        triton.cdiv(W_out, BLOCK_W),
    )

    _upsample_bicubic2d[grid](
        input,
        out_fp32,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        sN,
        sC,
        sH,
        sW,
        oN,
        oC,
        oH,
        oW,
        float(scale_h),
        float(scale_w),
        align_corners,
        BLOCK_W=BLOCK_W,
    )

    if input.dtype != torch.float32:
        return out_fp32.to(dtype=input.dtype)
    return out_fp32
