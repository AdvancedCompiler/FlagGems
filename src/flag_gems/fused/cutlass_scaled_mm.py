from typing import Optional

import torch
import triton
import triton.language as tl


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


def get_fp8_gemm_configs():
    configs = [
        {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        {
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_stages": 5,
            "num_warps": 2,
        },
        {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_stages": 5,
            "num_warps": 2,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        {
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
        {
            "BLOCK_M": 128,
            "BLOCK_N": 32,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 4,
        },
    ]
    return [
        triton.Config(c, num_stages=c.pop("num_stages"), num_warps=c.pop("num_warps"))
        for c in configs
    ]


@triton.jit
def grouped_launch(
    pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.autotune(
    configs=get_fp8_gemm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def block_scaled_mm_kernel(
    A,
    B,
    C,
    Bias,
    Ascale,
    Bscale,
    M,
    N,
    K,
    scale_block_n,
    scale_block_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_Ascale_m,
    stride_Ascale_k,
    stride_Bscale_k,
    stride_Bscale_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = grouped_launch(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    Ascale_ptrs = Ascale + offs_am * stride_Ascale_m

    offs_bsn = offs_bn // scale_block_n
    Bscale_ptrs = Bscale + offs_bsn * stride_Bscale_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K

        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        offs_ks = k * BLOCK_K // scale_block_k

        a_s = tl.load(Ascale_ptrs + offs_ks * stride_Ascale_k)
        b_s = tl.load(Bscale_ptrs + offs_ks * stride_Bscale_k)

        acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias_ptrs = Bias + offs_bn * stride_bias
        bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0)
        acc += bias[None, :]

    acc = acc.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)

    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def cutlass_scaled_mm(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape

    assert a.shape[0] == c.shape[0] and b.shape[1] == c.shape[1]
    assert a.shape[1] == b.shape[0]

    is_scale_a_vector = a_scales.dim() == 1 or (
        a_scales.dim() == 2 and a_scales.shape[1] == 1
    )

    is_scale_b_vector = b_scales.dim() == 1 or (
        b_scales.dim() == 2 and (b_scales.shape[0] == 1 or b_scales.shape[1] == 1)
    )

    if is_scale_a_vector and is_scale_b_vector:
        assert is_weak_contiguous(a)
        assert is_weak_contiguous(b)

        scale_a = a_scales.reshape(-1, 1) if a_scales.dim() <= 1 else a_scales
        scale_b = b_scales.reshape(-1, 1) if b_scales.dim() <= 1 else b_scales

        has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
        elif next_power_of_2_M <= 64:
            tile_shape = (64, 64, 256)
        elif next_power_of_2_M <= 128:
            tile_shape = (64, 128, 128)
        else:
            tile_shape = (128, 128, 128)

        block_size_m, block_size_n, block_size_k = tile_shape

        block_size_sa = 1 if has_scalar(scale_a) else block_size_m
        block_size_sb = 1 if has_scalar(scale_b) else block_size_n

        accumulator_dtype = tl.float32 if a.is_floating_point() else tl.int32

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        scaled_mm_kernel[grid](
            a,
            b,
            scale_a,
            scale_b,
            c,
            bias,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            accumulator_dtype,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
            BLOCK_SIZE_SCALE_A=block_size_sa,
            BLOCK_SIZE_SCALE_B=block_size_sb,
        )

        return c

    else:
        assert a.is_contiguous() and c.is_contiguous()
        assert a_scales.shape[0] == M
        scale_block_k = K // a_scales.shape[1]

        assert b_scales.shape[0] == a_scales.shape[1]
        scale_block_n = N // b_scales.shape[1]

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )

        block_scaled_mm_kernel[grid](
            a,
            b,
            c,
            bias,
            a_scales,
            b_scales,
            M,
            N,
            K,
            scale_block_n,
            scale_block_k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            bias.stride(0) if bias is not None else 0,
            a_scales.stride(0),
            a_scales.stride(1),
            b_scales.stride(0),
            b_scales.stride(1),
            HAS_BIAS=(bias is not None),
        )

        return c
