from tinygrad import Tensor
import math


def hann_window(x: int) -> Tensor:
    if x <= 1:
        return Tensor([1.0] * x)

    n = Tensor.arange(x)
    return 0.5 - 0.5 * (2 * math.pi * n / x).cos()
