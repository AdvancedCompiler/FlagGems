import logging

import torch
import triton
import triton.language as tl

from .. import runtime
from ..utils import dim_compress, libentry
from ..utils import triton_lang_extension as tle


@libentry()
@triton.autotune(configs=runtime.get_triton_config("index_select"), key=["M", "N"])
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)

    out_mask = rows_mask and (cols_offsets < index_len)

    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    selected = tl.load(inp + inp_off, mask=rows_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)


def index_select(inp, dim, index):
    logging.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()

    # with dim_compress
    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        return out.permute(order)
    else:
        return out

def cfggen():
    block_m = [1, 2, 4]
    block_n = [1024, 2048, 4096]
    configs = [
    triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=4)
        for m in block_m
        for n in block_n
    ]
    return configs

#kernel
@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def index_select_backward_kernel(
    grad, out, M, N, index, outN, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    
    grad_mask = rows_mask and (cols_offsets < N)
    out_mask = rows_mask and (cols_offsets < outN)
    
    indices = tl.load(index + cols_offsets, mask=(cols_offsets < N),other=0)
    grad_off = rows_offsets * N + cols_offsets[None, :]
    out_off = rows_offsets * outN + indices[None, :]
    selected = tl.load(grad + grad_off, mask=grad_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)

#function
def index_select_backward(grad, self_sizes,dim, index):
    logging.debug("GEMS INDEX SELECT BACKWARD")
    assert dim >= -len(self_sizes) and dim < len(self_sizes), "Invalid dim" 
    assert index.ndim <= 1, "Index should have dimension 1 or 0" 
    for i in index:
        assert (i >= 0 and i < self_sizes[dim]), "Index out of range"
    if index.ndim == 0:
        index = index.unsqueeze(0)
    index_shape = list(index.shape)
    dim = dim % len(self_sizes) 
    grad_shape = list(grad.shape)
    assert grad_shape[dim] == index_shape[0], "Index out of range"
    grad = dim_compress(grad, dim)
    N = grad_shape[dim]
    M = grad.numel() // N
    out_shape = list(grad.shape)
    out_shape[grad.ndim - 1] = self_sizes[dim]
    out = torch.zeros(out_shape, dtype=grad.dtype, device=grad.device)
    outN = out.shape[out.ndim-1]
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    index_select_backward_kernel[grid](grad, out, M, N, index, outN)
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        return out.permute(order)
    else:
        return out
