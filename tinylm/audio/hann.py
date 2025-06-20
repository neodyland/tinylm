from tinygrad import Tensor
import math


def hann_window(x: int) -> Tensor:
    if x <= 1:
        return Tensor([1.0] * x, requires_grad=False)

    n = Tensor.arange(x, requires_grad=False)
    return 0.5 - 0.5 * (2 * math.pi * n / x).cos()
