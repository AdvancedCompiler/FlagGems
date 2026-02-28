import pytest
import torch

import flag_gems
from benchmark.performance_utils import GenericBenchmark


def nll_loss_input_fn(shape, cur_dtype, device):
    N, C = shape[0], shape[1]
    extra_dims = shape[2:]

    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    inp = torch.nn.functional.log_softmax(inp, dim=1)

    target_shape = (N,) + extra_dims
    target = torch.randint(0, C, target_shape, dtype=torch.long, device=device)

    ignore_index = -100
    if target.numel() > 0:
        num_ignore = max(1, target.numel() // 10)
        flat = target.view(-1)
        flat[:num_ignore] = ignore_index
        target = flat.view(target_shape)

    weight = torch.rand(C, dtype=cur_dtype, device=device)

    for reduction in ["none", "mean", "sum"]:
        yield inp, target, {
            "weight": weight,
            "ignore_index": ignore_index,
            "reduction": reduction,
        }


class NllLossBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (32, 10),
            (4, 3, 5, 5),
            (4096, 32000),
            (8192, 1000),
        ]


@pytest.mark.nll_loss_benchmark
def test_perf_nll_loss():
    def torch_op(input, target, **kwargs):
        return torch.nn.functional.nll_loss(input, target, **kwargs)

    gems_op = flag_gems.nll_loss_nd
    bench = NllLossBenchmark(
        input_fn=nll_loss_input_fn,
        op_name="nll_loss",
        torch_op=torch_op,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )

    bench.set_gems(gems_op)

    bench.run()
