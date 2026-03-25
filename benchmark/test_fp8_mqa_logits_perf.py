import random
from itertools import product

import pytest
import torch
from vllm.utils.deep_gemm import fp8_mqa_logits as vllm_fp8_mqa_logits
from vllm.utils.import_utils import has_deep_gemm

import flag_gems
from benchmark.performance_utils import Benchmark
from flag_gems.ops.fp8_mqa_logits import fp8_mqa_logits as gems_fp8_mqa_logits

random.seed(42)


def is_vllm_available():
    try:
        return True
    except Exception:
        return False


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 90


VLLM_AVAILABLE = is_vllm_available()
DEEPGEMM_AVAILABLE = has_deep_gemm()
HOPPER_AVAILABLE = is_hopper_available()


def _build_case(M, H, D, N, q_dtype):
    q = torch.randn(
        (M, H, D),
        device=flag_gems.device,
        dtype=q_dtype,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)

    k = torch.randn(
        (N, D),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    k_fp8 = k.to(torch.float8_e4m3fn)
    k_scales = (
        torch.rand(
            (N,),
            device=flag_gems.device,
            dtype=torch.float32,
        )
        * 0.01
        + 0.001
    )

    weights = torch.randn(
        (M, H),
        device=flag_gems.device,
        dtype=torch.float32,
    )

    cu_seqlen_ks = torch.zeros(
        (M,),
        device=flag_gems.device,
        dtype=torch.int32,
    )
    cu_seqlen_ke = torch.full(
        (M,),
        N,
        device=flag_gems.device,
        dtype=torch.int32,
    )

    return (
        q_fp8,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )


class FP8MQALogitsCompareBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            "fp8_mqa_logits_gems_vs_deepgemm",
            self._vllm_wrapper,
            [torch.bfloat16],
        )
        self.set_gems(self._gems_wrapper)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, _dtype):
        test_shapes = [
            (32, 32, 128, 1024),
            (32, 32, 128, 2048),
            (32, 32, 128, 4096),
        ]
        q_dtypes = [torch.bfloat16, torch.float16]

        for (M, H, D, N), q_dtype in product(test_shapes, q_dtypes):
            case = _build_case(M, H, D, N, q_dtype)
            q_fp8, k_fp8, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke = case
            yield (
                q_fp8,
                k_fp8,
                k_scales,
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
                q_dtype,
            )

    @staticmethod
    def _vllm_wrapper(
        q_fp8,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        q_dtype,
    ):
        return vllm_fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scales),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=True,
        )

    @staticmethod
    def _gems_wrapper(
        q_fp8,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        q_dtype,
    ):
        return gems_fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scales),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=True,
        )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and VLLM_AVAILABLE
        and DEEPGEMM_AVAILABLE
        and HOPPER_AVAILABLE
    ),
    reason="requires CUDA + vLLM + DeepGEMM + Hopper",
)
@pytest.mark.performance
@pytest.mark.fp8_mqa_logits
def test_perf_fp8_mqa_logits_gems_vs_deepgemm():
    bench = FP8MQALogitsCompareBenchmark()
    bench.run()
