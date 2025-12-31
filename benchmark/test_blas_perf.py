import os
from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import (
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    BenchLevel,
    model_shapes,
)
from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark, GenericBenchmark2DOnly


class BlasBenchmark(Benchmark):
    """
    benchmark for blas
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, cur_dtype, self.device, False)

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            for b, m, n, k in self.shapes:
                yield from self.input_fn(b, m, n, k, cur_dtype, self.device, True)

    def set_more_shapes(self):
        large_k_shapes = [
            (8, 1848, 1536, 151936),
            (8, 1848, 1536, 128256),
            (8, 1848, 1536, 152064),
        ]

        model_shaps = model_shapes()
        return large_k_shapes + model_shaps

    def get_tflops(self, op, *args, **kwargs):
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        if self.op_name == "mm":
            total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2
        # shape(m,n)(n,p)
        # total_flops mxpx(2n+1)
        elif self.op_name == "addmm":
            total_flops = (
                args[0].shape[0] * args[1].shape[1] * (args[1].shape[0] * 2 + 1)
            )
        # total_flops bxnxpx2m
        elif self.op_name == "bmm":
            total_flops = (
                args[0].shape[0]
                * args[0].shape[1]
                * args[1].shape[2]
                * 2
                * args[0].shape[2]
            )
        return total_flops


class BaddbmmBenchmark(BlasBenchmark):
    """
    benchmark for Baddbmm
    """

    def set_more_shapes(self):
        model_shapes_list = model_shapes()

        skip_shapes = [
            (4, 8192, 128256, 4096),
            (4, 8192, 152064, 3584),
        ]

        filtered = []
        for shape in model_shapes_list:
            if shape not in skip_shapes:
                filtered.append(shape)

        return filtered

    def get_tflops(self, op, *args, **kwargs):
        # shape(b,m,k)(b,k,n)
        # total_flops = b * m * n * (2 * k + 1)
        total_flops = (
            args[1].shape[0]
            * args[1].shape[1]
            * args[2].shape[2]
            * (args[1].shape[2] * 2 + 1)
        )
        return total_flops


def addmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    bias = torch.randn([m, n], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2,


def bmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


def baddbmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device, requires_grad=True)

    if b_column_major:
        inp2 = torch.randn(
            [b, n, k], dtype=cur_dtype, device=device, requires_grad=True
        )
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn(
            [b, k, n], dtype=cur_dtype, device=device, requires_grad=True
        )

    bias = torch.randn([b, m, n], dtype=cur_dtype, device=device, requires_grad=True)

    yield bias, inp1, inp2


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, bench_cls",
    [
        pytest.param(
            "addmm",
            torch.addmm,
            addmm_input_fn,
            BlasBenchmark,
            marks=pytest.mark.addmm,
        ),
        pytest.param(
            "bmm",
            torch.bmm,
            bmm_input_fn,
            BlasBenchmark,
            marks=pytest.mark.bmm,
        ),
        pytest.param(
            "mm",
            torch.Tensor.mm,
            mm_input_fn,
            BlasBenchmark,
            marks=pytest.mark.mm,
        ),
        pytest.param(
            "baddbmm",
            torch.baddbmm,
            baddbmm_input_fn,
            BaddbmmBenchmark,
            marks=pytest.mark.baddbmm,
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn, bench_cls):
    if flag_gems.vendor_name == "mthreads" and op_name != "baddbmm":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    bench = bench_cls(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )
    bench.run()

    if flag_gems.vendor_name == "mthreads" and op_name != "baddbmm":
        del os.environ["MUSA_ENABLE_SQMMA"]


class MvAndOuterBenchmark(GenericBenchmark2DOnly):
    """
    Benchmark for MV and Outer operations
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


def mv_input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def outer_input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "mv",
            torch.Tensor.mv,
            mv_input_fn,
            marks=pytest.mark.mv,
        ),
        pytest.param(
            "outer",
            torch.Tensor.outer,
            outer_input_fn,
            marks=pytest.mark.outer,
        ),
    ],
)
def test_mv_and_outer_benchmark(op_name, torch_op, input_fn):
    bench = MvAndOuterBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class AddmvBenchmark(GenericBenchmark2DOnly):
    """
    Benchmark for addmv
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


def addmv_input_fn(m, n, cur_dtype, device):
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    bias = torch.randn([m], dtype=cur_dtype, device=device)
    # torch.addmv(bias, mat, vec)
    yield bias, mat, vec


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "addmv",
            torch.addmv,
            addmv_input_fn,
            marks=pytest.mark.addmv,
        ),
    ],
)
def test_addmv_benchmark(op_name, torch_op, input_fn):
    bench = AddmvBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class VdotBenchmark(BlasBenchmark):
    """
    benchmark for vdot
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            m = shape[0]
            yield from self.input_fn(m, cur_dtype, self.device)


@pytest.mark.vdot
def test_vdot_benchmark():
    def vdot_input_fn(m, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = VdotBenchmark(
        input_fn=vdot_input_fn,
        op_name="vdot",
        torch_op=torch.Tensor.vdot,
        dtypes=COMPLEX_DTYPES + FLOAT_DTYPES,
    )
    bench.run()


class AddrBenchmark(BlasBenchmark):
    """
    benchmark for addr
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            m, n = shape[0], shape[1]
            yield from self.input_fn(m, n, cur_dtype, self.device)


@pytest.mark.addr
def test_addr_benchmark():
    def addr_input_fn(m, n, cur_dtype, device):
        inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        inp3 = torch.randn([n], dtype=cur_dtype, device=device)
        yield inp1, inp2, inp3, {"alpha": 0.5, "beta": 0.5}

    bench = AddrBenchmark(
        input_fn=addr_input_fn,
        op_name="addr",
        torch_op=torch.Tensor.addr,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class CutlassScaledMMBenchmark(Benchmark):
    """
    benchmark for cutlass_scaled_mm
    """

    def __init__(self):
        out_dtypes = [torch.float16, torch.bfloat16]
        import vllm._custom_ops as ops  # noqa: F401

        super().__init__(
            "cutlass_scaled_mm", torch.ops._C.cutlass_scaled_mm, out_dtypes
        )
        self.set_gems(flag_gems.cutlass_scaled_mm)

    def set_more_shapes(self):
        self.shapes = []

        MNK_FACTORS = [
            (1, 256, 128),
            (1, 16384, 1024),
            (1, 24576, 496),
            (16, 256, 496),
            (16, 16384, 128),
            (16, 24576, 4096),
            (32, 8192, 4096),
            (32, 16384, 4096),
            (33, 1024, 1024),
            (33, 8192, 128),
            (64, 2048, 496),
            (64, 16384, 1024),
            (100, 8192, 496),
            (128, 32768, 4096),
            (256, 4096, 4096),
            (512, 256, 1024),
            (512, 8192, 4096),
            (512, 16384, 128),
            (512, 24576, 128),
        ]
        from random import seed, shuffle

        seed(42)

        shuffle(MNK_FACTORS)
        MNK_FACTORS = MNK_FACTORS[:16]
        if_use_bias = [True, False]
        dequantization_modes = ["Per-token", "Block-wise"]

        extended_shapes = []
        for shape in MNK_FACTORS:
            for use_bias in if_use_bias:
                for dequantization_mode in dequantization_modes:
                    extended_shapes.append((*shape, use_bias, dequantization_mode))

        shuffle(extended_shapes)

        return extended_shapes[:8]

    def get_input_iter(self, dtype):
        for M, N, K, use_bias, dequantization_mode in self.shapes:

            def to_int8(tensor: torch.Tensor):
                return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

            def to_fp8(tensor: torch.Tensor):
                finfo = torch.finfo(torch.float8_e4m3fn)
                return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
                    dtype=torch.float8_e4m3fn
                )

            if dequantization_mode == "Per-token":
                a = to_int8(torch.randn((M, K), device=flag_gems.device))
                b = to_int8(
                    torch.randn((K, N), device=flag_gems.device).t().contiguous().t()
                    * 5
                )
                scale_a = torch.randn((M,), device=flag_gems.device)
                scale_b = torch.randn((N,), device=flag_gems.device)
            else:
                from math import ceil

                a = to_fp8(torch.randn((M, K), device=flag_gems.device))
                b = to_fp8(
                    torch.randn((K, N), device=flag_gems.device).t().contiguous().t()
                )
                scale_a = torch.randn((M, ceil(K / 128)), device=flag_gems.device)
                scale_b = torch.randn(
                    (ceil(K / 128), ceil(N / 128)), device=flag_gems.device
                )
            bias = None
            output = torch.empty((a.shape[0], b.shape[1]), dtype=dtype, device=a.device)

            yield (output, a, b, scale_a, scale_b, bias)


def if_vllm_ok():
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False

    return True


@pytest.mark.skipif(not if_vllm_ok(), reason="vllm is not installed")
@pytest.mark.cutlass_scaled_mm
@pytest.mark.performance
def test_cutlass_scaled_mm_benchmark():
    bench = CutlassScaledMMBenchmark()
    bench.run()
