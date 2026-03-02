import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
    ],
    key=["total_elements"],
    reset_to_zero=["out_buffer_ptr"],
)
@triton.jit
def nll_loss_nd_kernel(
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
    REDUCTION: tl.constexpr,
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

    if REDUCTION == 0:
        tl.store(out_buffer_ptr + offsets, loss_val, mask=mask)

    else:
        block_loss_sum = tl.sum(loss_val, axis=0)

        if REDUCTION == 1:
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

    if reduction not in [0, 1, 2]:
        raise ValueError("reduction must be 0 ('none'), 1 ('mean'), or 2 ('sum')")

    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    if reduction == 0:
        # None
        out = torch.empty((N, S), device=input.device, dtype=torch.float32)

        nll_loss_nd_kernel[grid](
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
            REDUCTION=reduction,
        )

        if target.dim() == input.dim() - 1:
            res = out.view_as(target)
        else:
            res = out.reshape(target.shape)
        return res.to(input.dtype)

    else:
        if reduction == 1:
            out_buffer = torch.zeros(4, device=input.device, dtype=torch.float32)
        else:
            out_buffer = torch.zeros(1, device=input.device, dtype=torch.float32)

        nll_loss_nd_kernel[grid](
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
            REDUCTION=reduction,
        )

        out_val = out_buffer[3] if reduction == 1 else out_buffer[0]
        return out_val.to(input.dtype)
