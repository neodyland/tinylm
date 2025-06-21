import torchaudio
import torch
from ..testutil.torch_close import TorchTinygradCloseTest
import pytest
import numpy as np
from tinygrad import Tensor
from .melspectrogram import melspectrogram


class MelSpecTest(TorchTinygradCloseTest):
    def __init__(
        self,
        atol=0.00001,
        rtol=0.00001,
        power: float = 1.0,
        normalized: bool = False,
        n_fft: int = 4,
        hop_length: int = 1,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        sample_rate: int = 16000,
    ):
        super().__init__(atol, rtol)
        self.power = power
        self.normalized = normalized
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate

    def tinygrad(self, data: Tensor) -> Tensor:
        return melspectrogram(
            data,
            self.normalized,
            self.power,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            self.f_min,
            self.f_max,
            self.sample_rate,
        )

    def torch(self, data: torch.Tensor) -> torch.Tensor:
        melspec = torchaudio.transforms.MelSpectrogram(
            pad=0,
            window_fn=lambda n_fft: torch.hann_window(n_fft),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=self.normalized,
            center=True,
            onesided=None,
            power=self.power,
            win_length=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            sample_rate=self.sample_rate,
        )
        return melspec(data)


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("power", [1.0, 2.0])
@pytest.mark.parametrize("normalized", [False, True])
def test_melspec(n, power, normalized):
    test = MelSpecTest(power=power, normalized=normalized, n_fft=n, hop_length=n // 4)
    np.random.seed(42)
    data = np.random.randn(n).astype(np.float32)
    test.run(data)
