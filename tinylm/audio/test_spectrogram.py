from torchaudio import functional as AF
import torch
from ..testutil.torch_close import TorchTinygradCloseTest
import pytest
import numpy as np
from typing import Optional
from tinygrad import Tensor
from .spectrogram import spectrogram


class SpecTest(TorchTinygradCloseTest):
    def __init__(
        self,
        atol=0.00001,
        rtol=0.00001,
        power: Optional[float] = None,
        normalized: bool = False,
        n_fft: int = 4,
        hop_length: int = 1,
    ):
        super().__init__(atol, rtol)
        self.power = power
        self.normalized = normalized
        self.n_fft = n_fft
        self.hop_length = hop_length

    def tinygrad(self, data: Tensor) -> Tensor:
        return spectrogram(
            data, self.normalized, self.power, self.n_fft, self.hop_length
        )

    def torch(self, data: torch.Tensor) -> torch.Tensor:
        return AF.spectrogram(
            data,
            pad=0,
            window=torch.hann_window(self.n_fft),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=self.normalized,
            center=True,
            onesided=True,
            power=self.power,
            win_length=self.n_fft,
        )


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("power", [None, 1.0, 2.0])
@pytest.mark.parametrize("normalized", [False, True])
def test_spec(n, power, normalized):
    test = SpecTest(power=power, normalized=normalized, n_fft=n, hop_length=n // 4)
    np.random.seed(42)
    data = np.random.randn(n).astype(np.float32)
    test.run(data)
