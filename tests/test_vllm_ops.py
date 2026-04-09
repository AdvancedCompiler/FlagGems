import random
from itertools import product
from math import ceil
from typing import Optional

import pytest
import torch

import flag_gems
from flag_gems.utils.device_info import get_device_capability

from .accuracy_utils import gems_assert_close, to_reference
from .conftest import QUICK_MODE

try:
    import vllm  # noqa: F401
    from vllm.utils.deep_gemm import calc_diff
    from vllm.utils.deep_gemm import (
        fp8_paged_mqa_logits as fp8_paged_mqa_logits_deepgemm,
    )
    from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from vllm.utils.import_utils import has_deep_gemm

    VLLM_AVAILABLE = True
    DEEPGEMM_AVAILABLE = has_deep_gemm()
except ImportError:
    VLLM_AVAILABLE = False
    DEEPGEMM_AVAILABLE = False
    get_num_sms = None
    get_paged_mqa_logits_metadata = None
    calc_diff = None
    fp8_paged_mqa_logits_deepgemm = None


random.seed(42)


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


class CutlassScaledMMTestKit:
    num_test_cases = 16 if QUICK_MODE else 32

    @staticmethod
    def _get_all_combinations():
        # these shapes come from the test file of op `cutlass_scaled_mm` of vLLM
        mnk = [
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
        scale_shape_types = ["scalar", "vector", "matrix"]
        if_use_bias = [True, False]
        dtypes = [(torch.int8, torch.float16), (torch.float8_e4m3fn, torch.bfloat16)]

        combinations = product(
            mnk, scale_shape_types, scale_shape_types, if_use_bias, dtypes
        )
        return combinations

    @classmethod
    def _rand_sample(cls, all_params):
        random.shuffle(all_params)
        return all_params[: cls.num_test_cases]

    @classmethod
    def get_test_params(cls):
        combinations = cls._get_all_combinations()

        all_params = []
        for (
            (M, N, K),
            a_scale_category,
            b_scale_category,
            bias,
            (in_dtype, out_dtype),
        ) in combinations:
            is_scalar_or_vector_dequant = a_scale_category in [
                "scalar",
                "vector",
            ] and b_scale_category in ["scalar", "vector"]
            is_block_dequant = (
                a_scale_category == "matrix" and b_scale_category == "matrix"
            )

            if not (is_scalar_or_vector_dequant or is_block_dequant):
                continue

            if is_block_dequant and (bias is not None or M % 4 != 0):
                continue

            param = {
                "M": M,
                "N": N,
                "K": K,
                "a_scale_category": a_scale_category,
                "b_scale_category": b_scale_category,
                "use_bias": bias,
                "in_dtype": in_dtype,
                "out_dtype": out_dtype,
            }
            all_params.append(param)

        return cls._rand_sample(all_params)

    @staticmethod
    def get_scale_shape(M, N, K, category, is_a_scale=True):
        if category == "scalar":
            return (1,)
        elif category == "vector":
            if is_a_scale:
                return (M,)
            else:
                return (N,)
        else:
            if is_a_scale:
                return (M, ceil(K / 128))
            else:
                return (ceil(K / 128), ceil(N / 128))

    @staticmethod
    def baseline_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ):
        def group_broadcast(t: torch.Tensor, shape):
            for i, s in enumerate(shape):
                if t.shape[i] != s and t.shape[i] != 1:
                    assert s % t.shape[i] == 0
                    t = (
                        t.unsqueeze(i + 1)
                        .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                        .flatten(i, i + 1)
                    )
            return t

        scale_a_full = group_broadcast(scale_a, a.shape)
        scale_b_full = group_broadcast(scale_b, b.shape)

        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)

        lhs = scale_a_full * a_f32
        rhs = scale_b_full * b_f32

        output = torch.mm(lhs, rhs).to(out_dtype)

        if bias is not None:
            output = output + bias

        return output


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and CUDA_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture",
)
@pytest.mark.cutlass_scaled_mm
@pytest.mark.parametrize("p", CutlassScaledMMTestKit.get_test_params())
def test_cutlass_scaled_mm(p):
    kit = CutlassScaledMMTestKit

    M, N, K = p["M"], p["N"], p["K"]
    in_dtype = p["in_dtype"]
    out_dtype = p["out_dtype"]
    a_scale_category = p["a_scale_category"]
    b_scale_category = p["b_scale_category"]

    if in_dtype == torch.int8:
        a = to_int8(torch.randn((M, K), device=flag_gems.device))
        b = to_int8(
            torch.randn((K, N), device=flag_gems.device).t().contiguous().t() * 5
        )
    else:
        a = to_fp8(torch.randn((M, K), device=flag_gems.device))
        b = to_fp8(torch.randn((K, N), device=flag_gems.device).t().contiguous().t())

    a_scale_shape = kit.get_scale_shape(M, N, K, a_scale_category)
    b_scale_shape = kit.get_scale_shape(M, N, K, b_scale_category, False)

    scale_a = torch.randn(a_scale_shape, device=flag_gems.device, dtype=torch.float32)
    scale_b = torch.randn(b_scale_shape, device=flag_gems.device, dtype=torch.float32)

    scale_a = scale_a.contiguous()
    # convert scale_b to col-major
    # (for scalar/vector scale_b, this's a identical transformation)
    scale_b = scale_b.t().contiguous().t()

    bias = None
    if p["use_bias"]:
        bias = torch.randn((N,), device=flag_gems.device, dtype=out_dtype)

    c = torch.empty((M, N), device=flag_gems.device, dtype=out_dtype)

    flag_gems.cutlass_scaled_mm(c, a, b, scale_a, scale_b, bias)

    output_ref = kit.baseline_scaled_mm(
        a, b, scale_a.view(-1, 1), scale_b.view(1, -1), out_dtype, bias
    )

    if in_dtype == torch.int8:
        rtol, atol = 1e-1, 1.0
    else:
        rtol, atol = 5e-1, 1.5e-1

    torch.testing.assert_close(c, output_ref, rtol=rtol, atol=atol)


# ---------------------- fused_moe op test ----------------------
FUSED_MOE_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (8, 4, 64, 128, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if not QUICK_MODE:
    FUSED_MOE_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        (4, 16, 512, 1024, 2),
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (4, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
        (128, 8, 4096, 14336, 2),
        (256, 8, 4096, 14336, 2),
        (512, 8, 4096, 14336, 2),
        # DeepSeek-V3-like shapes (TP=8 shard)
        (1, 256, 7168, 2048, 8),
        (4, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
        (128, 256, 7168, 2048, 8),
        (256, 256, 7168, 2048, 8),
    ]


def torch_fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of fused MoE (no vLLM dependency).

    Computes:
        Y_m = sum_j  A_mj * W2[e_mj] @ SiLU(W1[e_mj] @ H_m)_{:D} ) * (W1[e_mj] @ H_m)_{D:})

    Args:
        hidden_states: (M, K)
        w1: (E, 2D, K)  -- gate + up projection concatenated
        w2: (E, K, D)   -- down projection
        topk_weights: (M, topk)
        topk_ids: (M, topk)

    Returns:
        output: (M, K)
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # GEMM1: up-projection  (1, K) @ (K, 2D) -> (1, 2D)
            z = hidden_states[m].to(torch.float32) @ w1[e].T.to(torch.float32)
            # SiLU-and-Mul: split into gate and up, apply SwiGLU
            D = z.shape[-1] // 2
            gate = z[:D]
            up = z[D:]
            s = (gate * torch.sigmoid(gate)) * up  # SiLU(gate) * up
            # GEMM2: down-projection  (1, D) @ (D, K) -> (1, K)
            r = s @ w2[e].T.to(torch.float32)
            # Weighted accumulation
            output[m] += (weight.to(torch.float32) * r).to(output.dtype)

    return output


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_accuracy_fused_moe_vs_ref(config, dtype):
    """Test FlagGems fused_moe against a pure PyTorch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    # Pure PyTorch reference (no vLLM dependency)
    ref = torch_fused_moe_reference(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    torch.cuda.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)

    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
def test_accuracy_fused_moe_vs_vllm(config, dtype):
    """Test FlagGems fused_moe against a pure PyTorch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    # Reference result
    ref = vllm_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
    )

    torch.cuda.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)

    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


FUSED_MOE_QUANT_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if not QUICK_MODE:
    FUSED_MOE_QUANT_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
    ]


def _fake_quantize_fp8(tensor: torch.Tensor):
    """Simulate FP8 E4M3 quantization round-trip for reference computation."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10
    # Per-tensor quantization
    amax = tensor.abs().amax().clamp(min=eps).float()
    scale = amax / fp8_max
    q = (tensor.float() / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    return q.float() * scale  # dequantized


def _fake_quantize_int8(tensor: torch.Tensor):
    """Simulate INT8 quantization round-trip for reference computation."""
    eps = 1e-10
    # Per-token quantization
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=eps).float()
    scale = amax / 127.0
    q = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q.float() * scale  # dequantized


def torch_fused_moe_quantized_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_mode: str = "fp8",
) -> torch.Tensor:
    """Reference fused MoE with simulated quantization noise.

    Simulates the quantization → dequantization round-trip on activations
    to model the same numerical behavior as the quantized kernel path.
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    fake_quant = _fake_quantize_fp8 if quant_mode == "fp8" else _fake_quantize_int8

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Quantize activation before GEMM1
            h_q = fake_quant(hidden_states[m].unsqueeze(0)).squeeze(0)
            # GEMM1
            z = h_q.float() @ w1[e].T.float()
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # Quantize intermediate before GEMM2
            s_q = fake_quant(s.unsqueeze(0)).squeeze(0)
            # GEMM2
            r = s_q.float() @ w2[e].T.float()
            output[m] += (weight.float() * r).to(output.dtype)

    return output


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    not is_cuda_available(),
    reason="FP8 quantization requires NVIDIA Hopper architecture",
)
def test_accuracy_fused_moe_fp8(config):
    """Test FlagGems fused_moe with FP8 W8A8 quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create FP8 weights: quantize and store scale
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    # Per-tensor quantization of weights
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10

    # Quantize w1 per-expert
    w1_scales = []
    w1_fp8_list = []
    for e in range(num_experts):
        amax = w1_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w1_q = (w1_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w1_fp8_list.append(w1_q)
        w1_scales.append(scale)
    w1_fp8 = torch.stack(w1_fp8_list)
    w1_scale = torch.tensor(w1_scales, device=device, dtype=torch.float32)

    # Quantize w2 per-expert
    w2_scales = []
    w2_fp8_list = []
    for e in range(num_experts):
        amax = w2_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w2_q = (w2_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w2_fp8_list.append(w2_q)
        w2_scales.append(scale)
    w2_fp8 = torch.stack(w2_fp8_list)
    w2_scale = torch.tensor(w2_scales, device=device, dtype=torch.float32)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems FP8 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference: use the dequantized weights (fp8 → float) for reference
    w1_deq = torch.zeros_like(w1_fp32).to(dtype)
    for e in range(num_experts):
        w1_deq[e] = (w1_fp8[e].float() * w1_scales[e]).to(dtype)
    w2_deq = torch.zeros_like(w2_fp32).to(dtype)
    for e in range(num_experts):
        w2_deq[e] = (w2_fp8[e].float() * w2_scales[e]).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states, w1_deq, w2_deq, topk_weights, topk_ids, quant_mode="fp8"
    )

    torch.cuda.synchronize()

    # FP8 quantization introduces more error than bf16, use wider tolerances.
    # Two quantized GEMMs + activation create cumulative rounding error.
    rtol = 5e-1
    atol = max(2e-1, ref.abs().max().item() * 1e-1)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
def test_accuracy_fused_moe_int8(config):
    """Test FlagGems fused_moe with INT8 W8A8 per-channel quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights: quantize per-channel (per output column of each expert)
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10

    # Per-channel quantization of weights: scale per [expert, output_dim]
    # w1 shape: [E, 2D, K] → scale shape: [E, 2D, 1]
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    # For the kernel: w1_scale shape [E, 2D] (per-channel: one scale per output dim)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT8 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference: use dequantized weights
    w1_deq = (w1_int8.float() * w1_scale_full).to(dtype)
    w2_deq = (w2_int8.float() * w2_scale_full).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states, w1_deq, w2_deq, topk_weights, topk_ids, quant_mode="int8"
    )

    torch.cuda.synchronize()

    # INT8 quantization introduces more error, use wider tolerances
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 2e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


def torch_fused_moe_weight_only_reference(
    hidden_states: torch.Tensor,
    w1_int: torch.Tensor,
    w2_int: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference fused MoE for weight-only quantization.

    Weights are dequantized (w_int * scale) then used in FP computation.
    Activations remain in original precision (no activation quantization).
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Dequantize weights
            w1_deq = w1_int[e].float() * w1_scale[e].unsqueeze(-1).float()
            w2_deq = w2_int[e].float() * w2_scale[e].unsqueeze(-1).float()
            # GEMM1
            z = hidden_states[m].float() @ w1_deq.T
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # GEMM2
            r = s @ w2_deq.T
            output[m] += (weight.float() * r).to(output.dtype)

    return output


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
def test_accuracy_fused_moe_int8_w8a16(config):
    """Test FlagGems fused_moe with INT8 W8A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    # Per-channel quantization
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)  # [E, 2D]

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)  # [E, K]

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT8 W8A16 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int8,
        w2_int8,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch.cuda.synchronize()

    # Weight-only quantization has less error than W8A8 since activations
    # are full precision, but still has weight quantization rounding error.
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 2e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
def test_accuracy_fused_moe_int4_w4a16(config):
    """Test FlagGems fused_moe with INT4 W4A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT4 weights stored in INT8 containers, per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    int4_max = 7
    int4_min = -8

    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / int4_max
    w1_int4 = (w1_fp32 / w1_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / int4_max
    w2_int4 = (w2_fp32 / w2_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT4 W4A16 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int4,
        w2_int4,
        topk_weights,
        topk_ids,
        use_int4_w4a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int4,
        w2_int4,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch.cuda.synchronize()

    # INT4 has coarser quantization → wider tolerance
    rtol = 3e-1
    atol = max(1e-1, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_moe
@pytest.mark.parametrize(
    "config",
    [
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_inplace(config, dtype):
    """Test that inplace=True writes output into hidden_states."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # Non-inplace reference
    ref = flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
    )

    # Inplace result
    hidden_copy = hidden_states.clone()
    result = flag_gems.fused_experts_impl(
        hidden_copy,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
    )

    torch.cuda.synchronize()

    # Result should be the same tensor as input
    assert result.data_ptr() == hidden_copy.data_ptr(), "inplace should reuse input"
    torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.fused_moe
@pytest.mark.parametrize(
    "config",
    [
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_apply_router_weight_on_input(config, dtype):
    """Test apply_router_weight_on_input vs default (weight on output)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # Default (weight on GEMM2 output)
    result_default = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
    )

    # Weight on GEMM1 input
    result_on_input = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=True,
    )

    torch.cuda.synchronize()

    # Due to SiLU nonlinearity, these will differ, but both should be
    # close to the reference with weight on the respective path.
    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    # The default (weight on output) should match our standard reference
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)
    torch.testing.assert_close(result_default, ref, rtol=rtol, atol=atol)

    # The apply_on_input result will differ but should be finite and nonzero
    assert torch.isfinite(
        result_on_input
    ).all(), "result_on_input has non-finite values"
    assert result_on_input.abs().sum() > 0, "result_on_input is all zeros"


try:
    from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from vllm.utils.import_utils import has_deep_gemm

    DEEPGEMM_AVAILABLE = has_deep_gemm()
except Exception:
    DEEPGEMM_AVAILABLE = False


@pytest.mark.get_paged_mqa_logits_metadata
@pytest.mark.skipif(not DEEPGEMM_AVAILABLE, reason="vllm with deep_gemm is required.")
@pytest.mark.parametrize("batch_size, next_n", [(4, 1), (2, 2)])
@pytest.mark.parametrize("avg_ctx_len", [1024, 2048])
def test_get_paged_mqa_logits_metadata(batch_size, next_n, avg_ctx_len):
    context_lens_2d = (
        torch.randint(
            int(0.8 * avg_ctx_len), int(1.2 * avg_ctx_len), (batch_size, next_n)
        )
        .cuda()
        .to(torch.int32)
    )

    ref = get_paged_mqa_logits_metadata(context_lens_2d, 64, get_num_sms())
    res = flag_gems.get_paged_mqa_logits_metadata(context_lens_2d, 64, get_num_sms())

    assert torch.equal(ref, res)


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf_bytes

    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def kv_cache_cast_to_fp8_triton(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    out = torch.empty(
        (num_blocks, block_size, num_heads, head_dim + 4),
        device=x.device,
        dtype=torch.uint8,
    )
    out[..., :head_dim] = x_scaled.view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    out[..., head_dim:] = sf_bytes.view(num_blocks, block_size, num_heads, 4)

    return out


def _build_mask(context_lens, batch_size, next_n, max_model_len, device):
    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    row_indices = torch.arange(batch_size * next_n, device=device) // next_n
    next_n_offset = torch.arange(batch_size * next_n, device=device) % next_n
    return positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(
        1
    )


@pytest.mark.fp8
@pytest.mark.paged_mqa_logits
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not DEEPGEMM_AVAILABLE, reason="DeepGEMM not available")
@pytest.mark.skipif(not is_cuda_available(), reason="SM90 and SM100 only")
@pytest.mark.parametrize("clean_logits", [True, False])
def test_accuracy_fp8_paged_mqa_logits(clean_logits: bool):
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    batch_size, next_n = 4, 1
    heads, index_dim = 32, 128
    avg_kv = 2048
    num_blocks, blocksize = max_model_len * 2, 64

    q = torch.randn(
        (batch_size, next_n, heads, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device=flag_gems.device, dtype=torch.float32
    )

    context_lens = torch.randint(
        int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,), device=flag_gems.device
    ).to(torch.int32)
    max_num_blocks_per_seq = (context_lens.max().item() + blocksize - 1) // blocksize
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), device=flag_gems.device, dtype=torch.int32
    )

    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        for j in range((ctx_len + blocksize - 1) // blocksize):
            block_tables[i][j] = block_idx_pool[counter]
            counter += 1

    q_fp8 = q.to(torch.float8_e4m3fn)

    kv_cache_fp8_deepgemm = kv_cache_cast_to_fp8(kv_cache)
    kv_cache_fp8_triton = kv_cache_cast_to_fp8_triton(kv_cache)

    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, blocksize, get_num_sms()
    )
    ref_out = fp8_paged_mqa_logits_deepgemm(
        q_fp8,
        kv_cache_fp8_deepgemm,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )
    ref_out = to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = flag_gems.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_triton,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    mask = _build_mask(
        context_lens, batch_size, next_n, max_model_len, flag_gems.device
    )

    ref_out = to_reference(ref_out)
    res_out = to_reference(res_out)
    mask = to_reference(mask)

    res_out_masked = torch.nan_to_num(res_out.masked_fill(~mask, 0), 0.0)
    ref_out_masked = torch.nan_to_num(ref_out.masked_fill(~mask, 0), 0.0)

    gems_assert_close(
        res_out_masked,
        ref_out_masked,
        res_out_masked.dtype,
        equal_nan=True,
        atol=5e-2,
        reduce_dim=1,
    )

    diff = calc_diff(res_out_masked, ref_out_masked)
    assert (
        diff < 1e-3
    ), f"Large discrepancy between Triton and DeepGEMM: {diff=} (expected < 1e-3)"


@pytest.mark.fp8
@pytest.mark.paged_mqa_logits
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not DEEPGEMM_AVAILABLE, reason="DeepGEMM not available")
@pytest.mark.skipif(not is_cuda_available(), reason="SM90 and SM100 only")
@pytest.mark.parametrize(
    "batch_size, next_n",
    [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (2, 2), (4, 2)],
)
@pytest.mark.parametrize(
    "heads, index_dim",
    [(16, 64), (32, 128)],
)
def test_accuracy_fp8_paged_mqa_logits_param(batch_size, next_n, heads, index_dim):
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    avg_kv = 2048
    num_blocks, blocksize = max_model_len * 2, 64

    q = torch.randn(
        (batch_size, next_n, heads, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device=flag_gems.device, dtype=torch.float32
    )

    context_lens = torch.randint(
        int(0.8 * avg_kv),
        int(1.2 * avg_kv),
        (batch_size,),
        device=flag_gems.device,
        dtype=torch.int32,
    )
    max_num_blocks_per_seq = (context_lens.max().item() + blocksize - 1) // blocksize
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), device=flag_gems.device, dtype=torch.int32
    )

    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        for j in range((ctx_len + blocksize - 1) // blocksize):
            block_tables[i][j] = block_idx_pool[counter]
            counter += 1

    q_fp8 = q.to(torch.float8_e4m3fn)

    kv_cache_fp8_deepgemm = kv_cache_cast_to_fp8(kv_cache)
    kv_cache_fp8_triton = kv_cache_cast_to_fp8_triton(kv_cache)

    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, blocksize, get_num_sms()
    )
    ref_out = fp8_paged_mqa_logits_deepgemm(
        q_fp8,
        kv_cache_fp8_deepgemm,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=True,
    )
    ref_out = to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = flag_gems.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_triton,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    mask = _build_mask(
        context_lens, batch_size, next_n, max_model_len, flag_gems.device
    )

    ref_out = to_reference(ref_out)
    res_out = to_reference(res_out)
    mask = to_reference(mask)

    res_out_masked = torch.nan_to_num(res_out.masked_fill(~mask, 0), 0.0)
    ref_out_masked = torch.nan_to_num(ref_out.masked_fill(~mask, 0), 0.0)

    gems_assert_close(
        res_out_masked,
        ref_out_masked,
        res_out_masked.dtype,
        equal_nan=True,
        atol=5e-2,
        reduce_dim=1,
    )

    diff = calc_diff(res_out_masked, ref_out_masked)
    assert (
        diff < 1e-3
    ), f"Large discrepancy between Triton and DeepGEMM: {diff=} (expected < 1e-3)"
