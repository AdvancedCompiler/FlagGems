import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log_sigmoid_forward(x):
    return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))


def log_sigmoid(x):
    logging.debug("GEMS LOG_SIGMOID FORWARD")

    return log_sigmoid_forward(x)
