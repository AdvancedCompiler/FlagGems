import torch
import triton
import triton.language as tl


@triton.jit
def topk_with_k2_triton(
    scores_ptr,
    bias_ptr,
    group_scores_ptr,
    num_experts_per_group,
    n_group,
    stride_scores_token,
    stride_bias_group,
    stride_group_scores_token,
    scoring_func: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    token_id = pid // n_group
    group_id = pid % n_group

    lane = tl.arange(0, BLOCK_SIZE)
    mask = lane < num_experts_per_group

    scores_offset = token_id * stride_scores_token + group_id * num_experts_per_group
    bias_offset = group_id * num_experts_per_group

    x = tl.load(
        scores_ptr + scores_offset + lane,
        mask=mask,
        other=-float("inf"),
    )

    b = tl.load(
        bias_ptr + bias_offset + lane,
        mask=mask,
        other=0.0,
    )

    x = x.to(tl.float32)
    if scoring_func == 1:  # sigmoid
        x = 1.0 / (1.0 + tl.exp(-x))

    x = x + b

    max1 = tl.max(x, axis=0)
    is_max1 = (x == max1) & mask
    count_max1 = tl.sum(is_max1, axis=0)

    x2 = tl.where(
        is_max1 & (count_max1 == 1),
        -float("inf"),
        x,
    )
    max2 = tl.max(x2, axis=0)

    group_scores_offset = token_id * stride_group_scores_token + group_id
    tl.store(
        group_scores_ptr + group_scores_offset,
        max1 + max2,
    )


@triton.jit
def group_idx_and_topk_triton(
    scores_ptr,
    group_scores_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    bias_ptr,
    num_tokens,
    n_group,
    topk_group,
    topk,
    num_experts,
    num_experts_per_group,
    renormalize: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    stride_scores_token,
    stride_group_scores_token,
    stride_out_token,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    neg_inf = -float("inf")
    valid_token = pid < num_tokens

    group_offsets = tl.arange(0, BLOCK_GROUP)
    valid_group = group_offsets < n_group

    group_scores = tl.load(
        group_scores_ptr + pid * stride_group_scores_token + group_offsets,
        mask=valid_group,
        other=neg_inf,
    )

    value = tl.where(valid_group, group_scores, neg_inf)
    target_num_min = BLOCK_GROUP - n_group + topk_group
    count_equal_to_top_value = BLOCK_GROUP - n_group
    pre_count_equal_to_top_value = 0
    topk_group_value = neg_inf

    for _ in range(TOPK_GROUP):
        need = count_equal_to_top_value < target_num_min
        max_val = tl.max(value, axis=0)

        mask = need & (value == max_val)
        value = tl.where(mask, neg_inf, value)

        newly = tl.sum(mask, axis=0).to(tl.int32)

        pre_count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value, pre_count_equal_to_top_value
        )
        count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value + newly, count_equal_to_top_value
        )

        topk_group_value = tl.where(need, max_val, topk_group_value)

    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value

    group_gt = group_scores > topk_group_value
    group_eq = group_scores == topk_group_value

    eq_i = group_eq.to(tl.int32)
    prefix_eq = tl.zeros([BLOCK_GROUP], dtype=tl.int32)

    prefix_eq = tl.cumsum(eq_i, axis=0) - eq_i

    group_selected = (
        group_gt | (group_eq & (prefix_eq < num_equalto_topkth_group))
    ) & valid_group

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    valid_expert = expert_offsets < num_experts
    expert_group = expert_offsets // num_experts_per_group  # [BLOCK_EXPERT]
    expert_in_group = expert_group[:, None] == group_offsets[None, :]

    expert_selected = (
        tl.sum(expert_in_group & group_selected[None, :], axis=1).to(tl.int1)
        & valid_expert
    )

    expert_scores = tl.load(
        scores_ptr + pid * stride_scores_token + expert_offsets,
        mask=expert_selected,
        other=-float("inf"),
    )

    if bias_ptr is not None:
        expert_scores += tl.load(
            bias_ptr + expert_offsets,
            mask=valid_expert,
            other=0.0,
        )

    expert_scores = tl.where(
        expert_selected,
        expert_scores,
        -float("inf"),
    )

    topk_vals = tl.full([TOPK], -float("inf"), tl.float32)
    topk_idx = tl.full([TOPK], 0, tl.int32)

    for i in range(TOPK):
        max_val = tl.max(expert_scores, axis=0)
        mask = expert_scores == max_val
        idx = tl.where(mask, expert_offsets, num_experts + 1)
        idx = tl.min(idx, axis=0)

        selected_score = tl.load(
            scores_ptr + pid * stride_scores_token + idx,
        )

        topk_vals = tl.where(
            tl.arange(0, TOPK) == i,
            selected_score,
            topk_vals,
        )
        topk_idx = tl.where(
            tl.arange(0, TOPK) == i,
            idx,
            topk_idx,
        )

        expert_scores = tl.where(expert_offsets == idx, -float("inf"), expert_scores)

        if renormalize:
            probs = tl.softmax(topk_vals)
            topk_vals = probs * routed_scaling_factor

    out_offsets = tl.arange(0, TOPK)

    tl.store(
        topk_values_ptr + pid * stride_out_token + out_offsets,
        topk_vals,
        mask=valid_token,
    )

    tl.store(
        topk_indices_ptr + pid * stride_out_token + out_offsets,
        topk_idx,
        mask=valid_token,
    )


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D Tensor")
    num_tokens, num_experts = scores.shape
    if num_experts % n_group != 0:
        raise ValueError("num_experts must be divisible by n_group")
    if n_group > 32:
        raise ValueError("n_group should be smaller than or equal to 32")
    if topk > 32:
        raise ValueError("topk should be smaller than or equal to 32 for now")
    # if scoring_func not in (SCORING_NONE, SCORING_SIGMOID):
    #     raise ValueError("scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)")

    num_experts_per_group = num_experts // n_group

    group_scores = torch.empty(
        (num_tokens, n_group),
        device=scores.device,
        dtype=scores.dtype,
    )

    topk_values = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.float32,
    )

    topk_indices = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.int32,
    )

    BLOCK1 = triton.next_power_of_2(num_experts_per_group)
    grid1 = (num_tokens * n_group,)

    topk_with_k2_triton[grid1](
        scores,
        bias,
        group_scores,
        num_experts_per_group,
        n_group,
        scores.stride(0),
        bias.stride(0),
        group_scores.stride(0),
        scoring_func,
        BLOCK_SIZE=BLOCK1,
    )

    BLOCK2 = triton.next_power_of_2(n_group)
    BLOCK_EXPERT = triton.next_power_of_2(num_experts)
    grid2 = (num_tokens,)

    group_idx_and_topk_triton[grid2](
        scores,
        group_scores,
        topk_values,
        topk_indices,
        bias,
        num_tokens,
        n_group,
        topk_group,
        topk,
        num_experts,
        num_experts_per_group,
        renormalize,
        routed_scaling_factor,
        scores.stride(0),
        group_scores.stride(0),
        topk_values.stride(0),
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        TOPK=topk,
        BLOCK_GROUP=BLOCK2,
        BLOCK_EXPERT=BLOCK_EXPERT,
    )
    return topk_values, topk_indices
