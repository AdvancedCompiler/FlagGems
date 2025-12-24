# test_simple.py
import itertools

import torch

from flag_gems import grouped_topk


def get_vllm_grouped_topk():
    try:
        from vllm._custom_ops import grouped_topk as vllm_grouped_topk

        return vllm_grouped_topk
    except (ImportError, AttributeError):
        raise RuntimeError(
            "❌ vLLM grouped_topk not available. "
            "Please ensure vLLM is installed with custom ops enabled."
        )


def run_one_case(
    scores,
    bias,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor=1.0,
    scoring_func=0,
):
    # device = scores.device
    vllm_grouped_topk = get_vllm_grouped_topk()

    # FlagGems
    fg_values, fg_indices = grouped_topk(
        scores=scores,
        n_group=n_group,
        topk_group=topk_group,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        bias=bias,
        scoring_func=scoring_func,
    )

    # vLLM
    vllm_values, vllm_indices = vllm_grouped_topk(
        scores=scores,
        num_expert_group=n_group,
        topk_group=topk_group,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        bias=bias,
        scoring_func=scoring_func,
    )

    # correctness check
    if not torch.allclose(fg_values, vllm_values, atol=1e-5, rtol=1e-5):
        raise AssertionError(
            f"❌ values mismatch\n" f"FlagGems:\n{fg_values}\n" f"vLLM:\n{vllm_values}"
        )

    if not torch.equal(fg_indices, vllm_indices):
        raise AssertionError(
            f"❌ indices mismatch\n"
            f"FlagGems:\n{fg_indices}\n"
            f"vLLM:\n{vllm_indices}"
        )


def run_tests():
    torch.manual_seed(45)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # test configurations
    # -------------------------------
    token_sizes = [1, 3, 8]
    expert_sizes = [8, 16]
    n_groups = [2, 4]
    topks = [1, 2]
    dtypes = [torch.bfloat16, torch.float32]
    # dtypes = [torch.bfloat16]

    total = 0
    passed = 0

    for (
        num_tokens,
        num_experts,
        n_group,
        topk,
        dtype,
    ) in itertools.product(
        token_sizes,
        expert_sizes,
        n_groups,
        topks,
        dtypes,
    ):
        scores = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
        # bias = torch.randn((num_experts,), dtype=torch.float32, device=device)#gems√ vllm×
        bias = torch.randn((num_experts,), dtype=dtype, device=device)

        print(
            f"\n=== Test case === "
            f"T={num_tokens}, E={num_experts}, "
            f"G={n_group}, topk={topk}, dtype={dtype}"
        )
        # print(f"scores:{scores}")
        # print(f"bias:{bias}")
        # print(f"scores+bias{scores + bias}")

        try:
            run_one_case(
                scores=scores,
                bias=bias,  # dtype=dtype时可以测试通过，dtype=torch.float32时dtypes=torch.bfloat16时大部分例子报错
                n_group=n_group,
                topk_group=topk,
                topk=topk,
                renormalize=False,  # todo:renormalize=True
                routed_scaling_factor=1.0,  # todo：renormalize=True对齐vllm后需测试不同的数值
                scoring_func=0,  # todo:scoring_func=1
            )
            print("✅ PASS")
            passed += 1
        except Exception:
            print("❌ FAIL")
            raise

        total += 1

    print(f"\n🎉 Summary: {passed}/{total} cases passed")


if __name__ == "__main__":
    run_tests()
