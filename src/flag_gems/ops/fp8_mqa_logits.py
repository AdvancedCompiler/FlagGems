import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fp8_mqa_logits_kernel(
    Q,
    K,
    K_SCALES,
    WEIGHTS,
    CU_SEQLEN_KS,
    CU_SEQLEN_KE,
    LOGITS,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CLEAN_LOGITS: tl.constexpr,
):
    """
    Triton kernel for FP8 MQA logits computation.

    Each program computes logits[m, n] = sum_h(ReLU(score[m, h, n]) * weights[m, h])
    where score[m, h, n] = sum_d(q[m, h, d] * k[n, d])
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= M or pid_n >= N:
        return

    ks_start = tl.load(CU_SEQLEN_KS + pid_m)
    ke_end = tl.load(CU_SEQLEN_KE + pid_m)

    n_valid = (pid_n >= ks_start) and (pid_n < ke_end)

    k_scale = tl.load(K_SCALES + pid_n)
    k_scale = k_scale.to(tl.float32)

    acc = 0.0

    for h_idx in range(H):
        weight = tl.load(WEIGHTS + pid_m * H + h_idx)
        weight = weight.to(tl.float32)

        score = 0.0

        for d_idx in range(0, D, BLOCK_D):
            block_size = tl.minimum(BLOCK_D, D - d_idx)

            q_ptr = Q + pid_m * H * D + h_idx * D + d_idx
            q = tl.load(
                q_ptr + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) < block_size,
                other=0.0,
            )
            q = q.to(tl.float32)

            k_ptr = K + pid_n * D + d_idx
            k = tl.load(
                k_ptr + tl.arange(0, BLOCK_D),
                mask=tl.arange(0, BLOCK_D) < block_size,
                other=0.0,
            )
            k = k.to(tl.float32) * k_scale

            score += tl.sum(q * k)

        score = tl.where(score > 0, score, 0.0)

        acc += score * weight

    if CLEAN_LOGITS:
        acc = tl.where(n_valid, acc, float("-inf"))

    logits_ptr = LOGITS + pid_m * N + pid_n
    tl.store(logits_ptr, acc)


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    logger.debug("FP8 MQA LOGITS")

    k_fp8, k_scales = kv

    M, H, D = q.shape
    N = k_fp8.shape[0]

    logits = torch.zeros((M, N), dtype=torch.float32, device=q.device)

    BLOCK_D = min(128, D)

    grid = (M, N)

    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        M,
        H,
        D,
        N,
        BLOCK_D,
        clean_logits,
    )

    return logits
