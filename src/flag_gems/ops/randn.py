import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

try:
    pair_uniform_to_normal = tl.pair_uniform_to_normal
except AttributeError:

    @triton.jit
    def pair_uniform_to_normal(u1, u2):
        """Box-Muller transform"""
        u1 = tl.maximum(1.0e-7, u1)
        th = 6.283185307179586 * u2
        r = tl.sqrt(-2.0 * tl.log(u1))
        return r * tl.cos(th), r * tl.sin(th)


device_ = device
logger = logging.getLogger(__name__)


# 每次 Philox/Box-Muller 迭代处理的元素数 (n0, n1, n2, n3)
UNROLL_PER_ITER = 4


@triton.autotune(
    configs=[
        # 尝试不同的 (线程块大小, 循环展开次数) 组合
        triton.Config({"BLOCK": 256, "UNROLL_FACTOR": 1}),
        triton.Config({"BLOCK": 512, "UNROLL_FACTOR": 1}),
        triton.Config({"BLOCK": 1024, "UNROLL_FACTOR": 1}),
        triton.Config({"BLOCK": 256, "UNROLL_FACTOR": 2}),
        triton.Config({"BLOCK": 512, "UNROLL_FACTOR": 2}),
        triton.Config({"BLOCK": 1024, "UNROLL_FACTOR": 2}),
        triton.Config({"BLOCK": 256, "UNROLL_FACTOR": 4}),
        triton.Config({"BLOCK": 512, "UNROLL_FACTOR": 4}),
        triton.Config({"BLOCK": 256, "UNROLL_FACTOR": 8}),
        triton.Config({"BLOCK": 512, "UNROLL_FACTOR": 8}),
    ],
    key=["N"],  # 基于总元素数量 N 来选择最佳配置
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,  # 新增的 Autotune 参数
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    # 此 Block (程序) 处理的基准计数器偏移
    # 每个 Block 处理 BLOCK * UNROLL_FACTOR * 4 个元素
    # 每个 Philox 计数器生成 4 个元素，所以我们需要 BLOCK * UNROLL_FACTOR 个计数器
    base_counter_offset = tl.program_id(0) * BLOCK * UNROLL_FACTOR

    # Block 内每个线程的偏移（向量）
    thread_offsets = tl.arange(0, BLOCK)

    # 此 Block (程序) 的基准存储偏移
    base_store_offset = tl.program_id(0) * BLOCK * UNROLL_FACTOR * UNROLL_PER_ITER

    # Philox 种子/密钥
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1_scalar = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    _O = c0_base * 0  # 一个 uint32 类型的 0 标量

    # 内核内循环
    for i in tl.static_range(UNROLL_FACTOR):
        # 1. 计算此迭代的 Philox 计数器（向量）
        #    c0_base: 基础偏移
        #    base_counter_offset: 此 Block 的偏移
        #    i * BLOCK: 此循环迭代的偏移
        #    thread_offsets: 此线程的偏移
        c0_i = c0_base + base_counter_offset + (i * BLOCK) + thread_offsets

        # 2. 生成 4 个 [0, 1) 均匀分布的向量
        r0, r1, r2, r3 = tl.philox(philox_seed, c0_i, c1_scalar, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)

        # 3. Box-Muller 变换（向量化）
        n0, n1 = pair_uniform_to_normal(r0, r1)
        n2, n3 = pair_uniform_to_normal(r2, r3)

        # 4. 计算此迭代的存储偏移（向量）
        iter_store_offset = base_store_offset + (i * BLOCK * UNROLL_PER_ITER)
        off_0 = iter_store_offset + thread_offsets
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        # 5. 存储 4 个正态分布的向量
        tl.store(out_ptr + off_0, n0, mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, n1, mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, n2, mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, n3, mask=off_3 < N, eviction_policy="evict_first")


def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS RANDN (Autotuned with Unroll)")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    if N == 0:
        return out

    # Grid (网格) 大小计算
    # 每个 Block (程序) 处理 BLOCK * UNROLL_FACTOR * UNROLL_PER_ITER 个元素
    grid_fn = lambda meta: (
        triton.cdiv(N, meta["BLOCK"] * meta["UNROLL_FACTOR"] * UNROLL_PER_ITER),
    )

    # Philox 增量基于总共需要生成的 (n0,n1,n2,n3) 四元组的数量
    increment = triton.cdiv(N, UNROLL_PER_ITER)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)

    with torch_device_fn.device(device):
        # Autotune 会自动找到最佳的 'BLOCK' 和 'UNROLL_FACTOR' 并传入
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out
