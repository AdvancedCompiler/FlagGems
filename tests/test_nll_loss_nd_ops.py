import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

# from Kernel import nll_loss


@pytest.mark.nll_loss_benchmark
@pytest.mark.parametrize(
    "N, C, extra_dims", [(32, 10, ()), (8, 5, (7,)), (4, 3, (5, 5))]
)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_nll_loss(N, C, extra_dims, reduction, dtype):
    # Initialize inputs
    shape = (N, C) + extra_dims
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    input = torch.nn.functional.log_softmax(input, dim=1)

    target_shape = (N,) + extra_dims
    target = torch.randint(
        0, C, target_shape, dtype=torch.long, device=flag_gems.device
    )

    # Set some positions to ignore_index
    ignore_index = -100
    if target.numel() > 0:
        num_ignore = max(1, target.numel() // 10)
        flat = target.view(-1)
        flat[:num_ignore] = ignore_index
        target = flat.view(target_shape)

    # Optional class weights
    weight = torch.rand(C, dtype=dtype, device=flag_gems.device)

    # Cast inputs for reference
    ref_input = to_reference(input, True)
    ref_weight = to_reference(weight, True)

    # Compute outputs
    ref_out = torch.nn.functional.nll_loss(
        ref_input, target, ref_weight, None, ignore_index, None, reduction
    )
    res_out = flag_gems.nll_loss_nd(
        input, target, weight, None, ignore_index, None, reduction
    )

    # Compare results
    reduce_dim = 1 if reduction == "none" else target.numel()
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
