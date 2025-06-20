import torch
from ..testutil.torch_close import TorchTinygradCloseTest
from tinygrad import Tensor
import pytest
from .stft import stft
from .hann import hann_window
import numpy as np


class STFTTest(TorchTinygradCloseTest):
    def tinygrad(self, data: Tensor) -> Tensor:
        window = hann_window(4)
        return stft(data, 4, window=window, hop_length=1)

    def torch(self, data: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(4)
        return torch.stft(data, 4, 1, return_complex=True, window=window, center=True)


@pytest.mark.parametrize("n", [4, 8])
def test_stft(n):
    test = STFTTest()
    np.random.seed(42)
    data = np.random.randn(n).astype(np.float32)
    test.run(data)
