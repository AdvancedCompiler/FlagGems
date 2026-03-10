import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def bincount_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(output_ptr + vals, 1, mask=mask)


@triton.jit
def bincount_weights_fp32_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)

    w_fp32 = w.to(tl.float32)
    tl.atomic_add(output_ptr + vals, w_fp32, mask=mask)


@triton.jit
def bincount_weights_fp64_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    w = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    w_fp64 = w.to(tl.float64)
    tl.atomic_add(output_ptr + vals, w_fp64, mask=mask)


def bincount(input, weights=None, minlength=0):
    logger.debug("GEMS BINCOUNT")

    assert input.dim() == 1, "input must be a 1-D tensor"
    assert minlength >= 0, "minlength must be non-negative"

    if weights is not None:
        assert weights.shape == input.shape, "weights must have the same shape as input"

    n = input.numel()

    if n == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=input.device)
        return torch.zeros(minlength, dtype=torch.int64, device=input.device)

    max_val = int(torch.max(input).item())
    output_size = max(max_val + 1, minlength)

    input_contig = input.contiguous()

    BLOCK_SIZE = 1024
    num_warps = 4
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    if weights is not None:
        weights_contig = weights.contiguous()
        out_dtype = weights.dtype

        if out_dtype == torch.float64:
            output = torch.zeros(output_size, dtype=torch.float64, device=input.device)
            bincount_weights_fp64_kernel[grid](
                input_contig,
                weights_contig,
                output,
                n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            output = torch.zeros(output_size, dtype=torch.float32, device=input.device)
            bincount_weights_fp32_kernel[grid](
                input_contig,
                weights_contig,
                output,
                n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
            if out_dtype != torch.float32:
                output = output.to(dtype=out_dtype)
    else:
        output = torch.zeros(output_size, dtype=torch.int64, device=input.device)
        bincount_kernel[grid](
            input_contig,
            output,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return output
