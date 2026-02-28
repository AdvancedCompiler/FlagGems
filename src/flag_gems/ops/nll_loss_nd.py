import math

import torch
import triton
import triton.language as tl


@triton.jit
def nll_loss_kernel_1(
    input_ptr,
    target_ptr,
    weight_ptr,
    out_loss_ptr,
    out_weight_ptr,
    total_elements,
    C,
    S,
    stride_in_n,
    stride_in_c,
    stride_in_s,
    stride_tgt_n,
    stride_tgt_s,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    n = offsets // S
    s = offsets % S

    tgt_offsets = n * stride_tgt_n + s * stride_tgt_s
    t = tl.load(target_ptr + tgt_offsets, mask=mask, other=ignore_index).to(tl.int32)

    valid = mask & (t != ignore_index) & (t >= 0) & (t < C)

    w = tl.load(weight_ptr + t, mask=valid, other=0.0).to(tl.float32)

    in_offsets = n * stride_in_n + t * stride_in_c + s * stride_in_s
    val = tl.load(input_ptr + in_offsets, mask=valid, other=0.0).to(tl.float32)

    loss_val = tl.where(valid, -val * w, 0.0)

    block_loss_sum = tl.sum(loss_val, axis=0)
    block_weight_sum = tl.sum(w, axis=0)

    tl.store(out_loss_ptr + pid, block_loss_sum)
    tl.store(out_weight_ptr + pid, block_weight_sum)


@triton.jit
def nll_loss_kernel_1_2(
    loss_scratch_ptr,
    weight_scratch_ptr,
    out_ptr,
    scratch_size,
    IS_MEAN: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < scratch_size

    loss_vals = tl.load(loss_scratch_ptr + offset, mask=mask, other=0.0)
    total_loss = tl.sum(loss_vals, axis=0)

    if IS_MEAN:
        weight_vals = tl.load(weight_scratch_ptr + offset, mask=mask, other=0.0)
        total_weight = tl.sum(weight_vals, axis=0)
        final_val = tl.where(total_weight == 0.0, 0.0, total_loss / total_weight)
    else:
        final_val = total_loss
    tl.store(out_ptr, final_val)


@triton.jit
def nll_loss_kernel_2(
    input_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    total_elements,
    C,
    S,
    stride_in_n,
    stride_in_c,
    stride_in_s,
    stride_tgt_n,
    stride_tgt_s,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    n = offsets // S
    s = offsets % S

    tgt_offsets = n * stride_tgt_n + s * stride_tgt_s
    t = tl.load(target_ptr + tgt_offsets, mask=mask, other=ignore_index).to(tl.int32)
    valid = mask & (t != ignore_index) & (t >= 0) & (t < C)

    in_offsets = n * stride_in_n + t * stride_in_c + s * stride_in_s
    logp = tl.load(input_ptr + in_offsets, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + t, mask=valid, other=0.0).to(tl.float32)

    loss = tl.where(valid, -logp * w, 0.0)

    tl.store(output_ptr + offsets, loss, mask=mask)


def nll_loss_nd(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    size_average: bool = None,
    ignore_index: int = -100,
    reduce: bool = None,
    reduction: str = "mean",
):
    if input.dim() < 2:
        raise ValueError("Input must have at least 2 dimensions (N, C, ...)")

    N = input.shape[0]
    C = input.shape[1]
    S = input.numel() // (N * C)

    inp = input.reshape(N, C, S)

    if target.dim() == 1 and S == 1:
        tgt = target.reshape(N, 1)
    elif target.numel() != N * S:
        raise ValueError(
            f"Target size {target.shape} doesn't match input size (N={N}, S={S})"
        )
    else:
        tgt = target.reshape(N, S)

    stride_in_n, stride_in_c, stride_in_s = inp.stride()
    stride_tgt_n, stride_tgt_s = tgt.stride()

    if weight is None:
        w = torch.ones(C, device=input.device, dtype=torch.float32)
    else:
        if weight.numel() != C:
            raise ValueError(f"Weight shape {weight.shape} must be ({C},)")
        w = weight.contiguous().to(torch.float32)

    total_elements = N * S

    reduction = reduction.lower()
    if reduction not in ["mean", "sum", "none"]:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    if reduction in ["mean", "sum"]:
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(total_elements)))
        scratch_size = triton.cdiv(total_elements, block_size)
        block_mid = triton.next_power_of_2(scratch_size)

        loss_scratch = torch.empty(
            (scratch_size,), device=input.device, dtype=torch.float32
        )
        weight_scratch = torch.empty(
            (scratch_size,), device=input.device, dtype=torch.float32
        )

        nll_loss_kernel_1[(scratch_size,)](
            inp,
            tgt,
            w,
            loss_scratch,
            weight_scratch,
            total_elements,
            C,
            S,
            stride_in_n,
            stride_in_c,
            stride_in_s,
            stride_tgt_n,
            stride_tgt_s,
            ignore_index,
            BLOCK_SIZE=block_size,
        )

        out = torch.empty([], device=input.device, dtype=torch.float32)
        is_mean = reduction == "mean"

        nll_loss_kernel_1_2[(1,)](
            loss_scratch,
            weight_scratch,
            out,
            scratch_size,
            IS_MEAN=is_mean,
            BLOCK_MID=block_mid,
        )
        return out.to(input.dtype)

    else:
        BLOCK_SIZE = 1024
        grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
        out = torch.empty((N, S), device=input.device, dtype=torch.float32)

        nll_loss_kernel_2[(grid_size,)](
            inp,
            tgt,
            w,
            out,
            total_elements,
            C,
            S,
            stride_in_n,
            stride_in_c,
            stride_in_s,
            stride_tgt_n,
            stride_tgt_s,
            ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if target.dim() == input.dim() - 1:
            res = out.view_as(target)
        else:
            res = out.reshape(target.shape)
        return res.to(input.dtype)
