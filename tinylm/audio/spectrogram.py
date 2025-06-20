from tinygrad import Tensor
from .stft import stft
from .hann import hann_window
from typing import Optional


def spectrogram(
    x: Tensor, normalized: bool, power: Optional[float], n_fft: int, hop_length: int
) -> Tensor:
    window = hann_window(n_fft)
    r = stft(x, n_fft=n_fft, hop_length=hop_length, window=window)
    if normalized:
        r /= window.pow(2.0).sum().sqrt()
    if power is not None:
        r_real = r[:, 0, :].abs()
        r_imag = r[:, 1, :].abs()
        r = (r_real**2 + r_imag**2).sqrt()
        if power == 1.0:
            return r
        return r.pow(power)
    return r
