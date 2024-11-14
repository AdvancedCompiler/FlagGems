import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry


def cfggen_first():
    block_m = [1, 2, 4, 8, 32]
    block_n = [512, 1024, 2048]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
    ]
    return configs


def cfggen_last():
    block_m = [512, 1024, 2048]
    block_n = [1, 2, 4, 8, 32]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
    ]
    return configs


def cfggen():
    block_m = [1, 2, 4, 8, 32]
    block_n = [256, 512, 1024, 2048]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen_last(), key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_last(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = tl.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = bx + tx
    col_mask = col_offset < N

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * v_value

    normalized = tl.sqrt(tl.sum(v_block, axis=1) + eps)
    tl.store(norm + col_offset, normalized[:, None], mask=col_mask)
    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)

    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_vec = v_value / normalized[:, None]
        out = v_vec * g_value
        tl.store(output + row_offset * N + col_offset, out, mask=mask)


@libentry()
@triton.autotune(configs=cfggen_first(), key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def weight_norm_kernel_first(
    output,
    norm,
    v,
    g,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = tl.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = by + ty
    row_mask = row_offset < M

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * v_value

    normalized = tl.sqrt(tl.sum(v_block, axis=1) + eps)
    tl.store(norm + row_offset, normalized[:, None], mask=row_mask)
    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)

    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_vec = v_value / normalized[:, None]
        out = v_vec * g_value
        tl.store(output + row_offset * N + col_offset, out, mask=mask)


@libentry()
@triton.autotune(configs=cfggen_last(), key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_last(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tx = tl.arange(0, BLOCK_COL_SIZE)[:, None]
    bx = tl.program_id(axis=0) * BLOCK_COL_SIZE
    col_offset = tx + bx
    col_mask = col_offset < N

    g_value = tl.load(g + col_offset, mask=col_mask).to(tl.float32)
    norm_value = tl.load(norm + col_offset, mask=col_mask).to(tl.float32)

    ty = tl.arange(0, BLOCK_ROW_SIZE)[None, :]

    vw_block = tl.zeros([BLOCK_COL_SIZE, BLOCK_ROW_SIZE], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        vw_block += v_value * w_value
    vw_sum = tl.sum(vw_block, 1)[:, None]

    for base in range(0, M, BLOCK_ROW_SIZE):
        row_offset = base + ty
        mask = row_offset < M and col_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + col_offset, g_grad_value, mask=col_mask)


@libentry()
@triton.autotune(configs=cfggen_first(), key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def weight_norm_bwd_kernel_first(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    ty = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    by = tl.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = by + ty
    row_mask = row_offset < M

    g_value = tl.load(g + row_offset, mask=row_mask).to(tl.float32)
    norm_value = tl.load(norm + row_offset, mask=row_mask).to(tl.float32)

    tx = tl.arange(0, BLOCK_COL_SIZE)[None, :]

    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_block += v_value * w_value
    vw_sum = tl.sum(v_block, 1)[:, None]

    for base in range(0, N, BLOCK_COL_SIZE):
        col_offset = base + tx
        mask = col_offset < N and row_mask
        v_value = tl.load(v + row_offset * N + col_offset, mask=mask).to(tl.float32)
        w_value = tl.load(w + row_offset * N + col_offset, mask=mask).to(tl.float32)
        v_grad_value = g_value * (
            w_value / (norm_value + eps)
            - v_value / (norm_value * norm_value * norm_value + eps) * vw_sum
        )
        tl.store(v_grad + row_offset * N + col_offset, v_grad_value, mask=mask)

    g_grad_value = vw_sum / (norm_value + eps)
    tl.store(g_grad + row_offset, g_grad_value, mask=row_mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["v_shape0", "v_shape1", "v_shape2"])
@triton.jit(do_not_specialize=["eps"])
def norm_kernel(
    output,
    v,
    v_shape0,
    v_shape1,
    v_shape2,
    eps: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tid_m = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    pid = tl.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = pid + tid_m
    row_mask = row_offset < v_shape1

    tid_n = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    v_block = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for base in range(0, v_shape0 * v_shape2, BLOCK_COL_SIZE):
        col_offset = base + tid_n
        m_idx = col_offset // v_shape2
        n_idx = row_offset
        k_idx = col_offset % v_shape2

        mask = m_idx < v_shape0 and row_mask

        v_offsets = m_idx * v_shape1 * v_shape2 + n_idx * v_shape2 + k_idx
        v_value = tl.load(v + v_offsets, mask=mask)
        v_block += v_value * v_value
    v_sum = tl.sum(v_block, axis=1) + eps
    v_norm = tl.sqrt(v_sum[:, None])
    tl.store(output + row_offset, v_norm, mask=row_mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["v_shape0", "v_shape1", "v_shape2"])
@triton.jit(do_not_specialize=["eps"])
def norm_bwd_kernel(
    v_grad,
    norm_grad,
    v,
    norm,
    v_shape0,
    v_shape1,
    v_shape2,
    eps: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    tid_m = tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    pid = tl.program_id(axis=0) * BLOCK_ROW_SIZE
    row_offset = pid + tid_m
    row_mask = row_offset < v_shape1

    norm_value = tl.load(norm + row_offset, mask=row_mask)
    norm_grad_value = tl.load(norm_grad + row_offset, mask=row_mask)

    tid_n = tl.arange(0, BLOCK_COL_SIZE)[None, :]
    for base in range(0, v_shape0 * v_shape2, BLOCK_COL_SIZE):
        col_offset = base + tid_n
        m_idx = col_offset // v_shape2
        n_idx = row_offset
        k_idx = col_offset % v_shape2

        mask = m_idx < v_shape0 and row_mask

        v_offsets = m_idx * v_shape1 * v_shape2 + n_idx * v_shape2 + k_idx
        v_value = tl.load(v + v_offsets, mask=mask)
        v_grad_value = v_value / norm_value * norm_grad_value
        tl.store(v_grad + v_offsets, v_grad_value, mask=mask)


class WeightNormInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim):
        logging.debug("GEMS WEIGHTNORM FORWARD")
        v = v.contiguous()
        g = g.contiguous()
        output = torch.empty_like(v)
        norm = torch.empty_like(g, dtype=torch.float32)
        if dim == 0:
            M = v.shape[0]
            N = math.prod(v.shape[1:])
            grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
            with torch.cuda.device(v.device):
                weight_norm_kernel_first[grid](
                    output, norm, v, g, M, N, eps=torch.finfo(torch.float32).tiny
                )
        elif dim == v.ndim - 1:
            M = math.prod(v.shape[:-1])
            N = v.shape[dim]
            grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
            with torch.cuda.device(v.device):
                weight_norm_kernel_last[grid](
                    output, norm, v, g, M, N, eps=torch.finfo(torch.float32).tiny
                )
        ctx.save_for_backward(v, g, norm)
        ctx.DIM = dim
        return output, norm

    @staticmethod
    def backward(ctx, w_grad, norm_grad):
        logging.debug("GEMS WEIGHTNORM BACKWARD")
        v, g, norm = ctx.saved_tensors
        dim = ctx.DIM
        w_grad = w_grad.contiguous()
        norm_grad = norm_grad.contiguous()
        v_grad = torch.empty_like(v)
        g_grad = torch.empty_like(g)

        if dim == 0:
            M = v.shape[0]
            N = math.prod(v.shape[1:])
            grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)
            with torch.cuda.device(v.device):
                weight_norm_bwd_kernel_first[grid](
                    v_grad,
                    g_grad,
                    w_grad,
                    v,
                    g,
                    norm,
                    M,
                    N,
                    eps=torch.finfo(torch.float32).tiny,
                )
        elif dim == v.ndim - 1:
            M = math.prod(v.shape[:dim])
            N = v.shape[dim]
            grid = lambda META: (triton.cdiv(N, META["BLOCK_COL_SIZE"]),)
            with torch.cuda.device(v.device):
                weight_norm_bwd_kernel_last[grid](
                    v_grad,
                    g_grad,
                    w_grad,
                    v,
                    g,
                    norm,
                    M,
                    N,
                    eps=torch.finfo(torch.float32).tiny,
                )
        return v_grad, g_grad, None


class Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, dim):
        logging.debug("GEMS NORM FORWARD")
        v = v.contiguous()
        output = torch.empty(
            *[1 if i != dim else v.shape[dim] for i in range(v.ndim)],
            dtype=v.dtype,
            device=v.device,
        )
        v_shape = [
            math.prod(v.shape[:dim]),
            v.shape[dim],
            math.prod(v.shape[dim + 1 :]),
        ]

        grid = lambda META: (triton.cdiv(v_shape[1], META["BLOCK_ROW_SIZE"]),)

        with torch.cuda.device(v.device):
            norm_kernel[grid](
                output,
                v,
                v_shape[0],
                v_shape[1],
                v_shape[2],
                eps=torch.finfo(torch.float32).tiny,
            )
        ctx.save_for_backward(v, output)
        ctx.V_SHAPE = v_shape
        return output

    @staticmethod
    def backward(ctx, norm_grad):
        logging.debug("GEMS NORM BACKWARD")
        norm_grad = norm_grad.contiguous()
        v, norm = ctx.saved_tensors
        v_grad = torch.empty_like(v)

        grid = lambda META: (triton.cdiv(ctx.V_SHAPE[1], META["BLOCK_ROW_SIZE"]),)
        with torch.cuda.device(v.device):
            norm_bwd_kernel[grid](
                v_grad,
                norm_grad,
                v,
                norm,
                ctx.V_SHAPE[0],
                ctx.V_SHAPE[1],
                ctx.V_SHAPE[2],
                eps=torch.finfo(torch.float32).tiny,
            )
        return v_grad, None


def weight_norm_interface(v, g, dim=0):
    return WeightNormInterface.apply(v, g, dim)


def weight_norm(v, g, dim=0):
    dim = dim % v.ndim
    has_half_dtype = v.dtype == torch.float16 or g.dtype == torch.float16
    can_use_fused = (not has_half_dtype) and (dim == 0 or dim == v.ndim - 1)
    if can_use_fused:
        output, _ = weight_norm_interface(v, g, dim)
        return output
    else:
        return v * (g / Norm.apply(v, dim))