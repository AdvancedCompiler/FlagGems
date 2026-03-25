import random

import pytest
import torch
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_mqa_logits

import flag_gems

from .accuracy_utils import gems_assert_close

device = flag_gems.device


@pytest.mark.fp8_mqa_logits
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.has_device_capability(90), reason="SM90+ only")
@pytest.mark.parametrize("clean_logits", [True, False])
def test_accuracy_fp8_mqa_logits(clean_logits: bool):
    torch.manual_seed(0)
    random.seed(0)

    M = 4
    H = 32
    D = 128
    N = 4096

    q = torch.randn((M, H, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_fp8 = torch.randn((N, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_scales = torch.rand((N,), device=device, dtype=torch.float32) * 0.01 + 0.001
    weights = torch.randn((M, H), device=device, dtype=torch.float32)

    cu_seqlen_ks = torch.tensor([0, 1024, 2048, 3072], device=device, dtype=torch.int32)
    cu_seqlen_ke = torch.tensor(
        [1024, 2048, 3072, 4096], device=device, dtype=torch.int32
    )

    ref_out = fp8_mqa_logits(
        q, (k_fp8, k_scales), weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )

    with flag_gems.use_gems():
        res_out = fp8_mqa_logits(
            q=q,
            kv=(k_fp8, k_scales),
            weights=weights,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            clean_logits=clean_logits,
        )

    gems_assert_close(
        res_out, ref_out, res_out.dtype, equal_nan=True, atol=5e-2, reduce_dim=1
    )


@pytest.mark.fp8_mqa_logits
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not current_platform.has_device_capability(90), reason="SM90+ only")
@pytest.mark.parametrize("M, N", [(32, 2048), (32, 4096), (32, 1024)])
@pytest.mark.parametrize("H, D", [(32, 128)])
def test_accuracy_fp8_mqa_logits_param(M: int, N: int, H: int, D: int):
    torch.manual_seed(0)
    random.seed(0)
    clean_logits = True

    q = torch.randn((M, H, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_fp8 = torch.randn((N, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_scales = torch.rand((N,), device=device, dtype=torch.float32) * 0.01 + 0.001
    weights = torch.randn((M, H), device=device, dtype=torch.float32)

    cu_seqlen_ks = torch.zeros(M, device=device, dtype=torch.int32)
    cu_seqlen_ke = torch.ones(M, device=device, dtype=torch.int32) * N
    for m in range(M):
        cu_seqlen_ks[m] = m * (N // M // 2)
        cu_seqlen_ke[m] = N - (M - m - 1) * (N // M // 2)

    ref_out = fp8_mqa_logits(
        q, (k_fp8, k_scales), weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )

    with flag_gems.use_gems():
        res_out = fp8_mqa_logits(
            q=q,
            kv=(k_fp8, k_scales),
            weights=weights,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            clean_logits=clean_logits,
        )

    gems_assert_close(
        res_out, ref_out, res_out.dtype, equal_nan=True, atol=5e-2, reduce_dim=1
    )
