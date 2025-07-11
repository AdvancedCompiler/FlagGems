import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0)


def relu(self):
    logger.debug("GEMS RELU FORWARD")
    output = relu_forward(self)
    return output


def relu_(A):
    logger.debug("GEMS RELU_ FORWARD")
    out = relu_forward(A, out0=A)
    return out
