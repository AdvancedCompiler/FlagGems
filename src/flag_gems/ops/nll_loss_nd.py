import torch
import triton
import triton.language as tl


@triton.jit
def nll_loss_kernel_1(
    input_ptr,
    target_ptr,
    weight_ptr,
    out_buffer_ptr,
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
    ONE_BLOCK: tl.constexpr,
    IS_MEAN: tl.constexpr,
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

    if ONE_BLOCK:
        if IS_MEAN:
            block_weight_sum = tl.sum(w, axis=0)
            final_val = tl.where(
                block_weight_sum == 0.0, 0.0, block_loss_sum / block_weight_sum
            )
        else:
            final_val = block_loss_sum
        tl.store(out_buffer_ptr, final_val)
    else:
        if IS_MEAN:
            block_weight_sum = tl.sum(w, axis=0)

            tl.atomic_add(out_buffer_ptr + 0, block_loss_sum, sem="relaxed")
            tl.atomic_add(out_buffer_ptr + 1, block_weight_sum, sem="relaxed")

            old_cnt = tl.atomic_add(out_buffer_ptr + 2, 1.0, sem="release")

            if old_cnt == tl.num_programs(0) - 1.0:
                total_loss = tl.load(out_buffer_ptr + 0)
                total_weight = tl.load(out_buffer_ptr + 1)
                final_val = tl.where(
                    total_weight == 0.0, 0.0, total_loss / total_weight
                )

                tl.store(out_buffer_ptr + 3, final_val)
        else:
            tl.atomic_add(out_buffer_ptr + 0, block_loss_sum, sem="relaxed")


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
    reduction: int = 1,
    ignore_index: int = -100,
):
    if input.dim() < 3:
        raise ValueError(
            "nll_loss_nd requires input to have at least 3 dimensions (N, C, d1, ..., dK)"
        )

    N = input.shape[0]
    C = input.shape[1]
    S = input.numel() // (N * C)

    inp = input.reshape(N, C, S)

    if target.numel() != N * S:
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

    if total_elements <= 1024:
        BLOCK_SIZE = max(32, triton.next_power_of_2(total_elements))
    else:
        BLOCK_SIZE = 1024

    if BLOCK_SIZE <= 64:
        num_warps = 2
    elif BLOCK_SIZE <= 256:
        num_warps = 4
    else:
        num_warps = 8

    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)

    if reduction not in [0, 1, 2]:
        raise ValueError("reduction must be 0 ('none'), 1 ('mean'), or 2 ('sum')")

    if reduction in [1, 2]:
        is_mean = reduction == 1

        if grid_size == 1:
            out = torch.empty([], device=input.device, dtype=torch.float32)
            nll_loss_kernel_1[(1,)](
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
                ONE_BLOCK=True,
                IS_MEAN=is_mean,
                num_warps=num_warps,
            )
            return out.to(input.dtype)
        else:
            if is_mean:
                out_buffer = torch.zeros(4, device=input.device, dtype=torch.float32)
            else:
                out_buffer = torch.zeros(1, device=input.device, dtype=torch.float32)

            nll_loss_kernel_1[(grid_size,)](
                inp,
                tgt,
                w,
                out_buffer,
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
                ONE_BLOCK=False,
                IS_MEAN=is_mean,
                num_warps=num_warps,
            )

            out_val = out_buffer[3] if is_mean else out_buffer[0]
            return out_val.to(input.dtype)

    else:
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
            num_warps=num_warps,
        )

        if target.dim() == input.dim() - 1:
            res = out.view_as(target)
        else:
            res = out.reshape(target.shape)
        return res.to(input.dtype)
