import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit(do_not_specialize=["numel", "tokens_per_thread"])
def count_and_sort_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def count_and_sort_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def count_and_sort_stage3(
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + token_cnt
        tl.store(cumsum_ptr + i, last_cumsum)


@triton.jit(do_not_specialize=["numel", "tokens_per_thread"])
def count_and_sort_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
):
    pid = tl.program_id(0)

    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx):
        tl.store(expert_ids_ptr + i, pid)

    start_idx_t = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx_t, tl.minimum(start_idx_t + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        base_offset = tl.load(cumsum_ptr + expert_id)
        rank_post_pad = token_cnt + base_offset

        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


def count_and_sort_expert_tokens_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)

    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)

    tokens_per_thread = ceil_div(numel, num_experts)

    count_and_sort_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    count_and_sort_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    count_and_sort_stage3[(1,)](
        tokens_cnts,
        cumsum,
        num_experts,
    )
    count_and_sort_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        numel,
        tokens_per_thread,
    )


def count_and_sort_expert_tokens(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Implements vllm::moe::count_and_sort_expert_tokens_kernel logic via Triton.
    Sorts tokens by expert ID.
    """

    sorted_token_ids = torch.empty_like(topk_ids.flatten(), dtype=torch.int32)
    expert_ids = torch.empty_like(topk_ids.flatten(), dtype=torch.int32)

    count_and_sort_expert_tokens_triton(
        topk_ids.flatten(),
        num_experts,
        sorted_token_ids,
        expert_ids,
    )
    return sorted_token_ids, expert_ids
