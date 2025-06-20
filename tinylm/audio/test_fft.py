import torch
import numpy as np
import pytest
from tinygrad import Tensor
from .fft import fft
from ..testutil.torch_close import TorchTinygradCloseTest


class FFTTest(TorchTinygradCloseTest):
    def tinygrad(self, data: Tensor) -> Tensor:
        return fft(data)

    def torch(self, data: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(data)


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_fft(n):
    test = FFTTest()
    np.random.seed(42)
    data = np.random.randn(n).astype(np.float32)
    test.run(data)
