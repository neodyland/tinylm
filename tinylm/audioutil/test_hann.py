import torch
from ..testutil.torch_close import TorchTinygradCloseTest
from tinygrad import Tensor
import pytest
from .hann import hann_window


class HannTest(TorchTinygradCloseTest):
    def tinygrad(self, data: int) -> Tensor:
        return hann_window(data)

    def torch(self, data: int) -> torch.Tensor:
        return torch.hann_window(data)


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_hann(n):
    test = HannTest()
    test.run(n)
