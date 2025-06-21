from tinygrad import Tensor
from .spectrogram import spectrogram
from typing import Optional, Literal
import math

type MelScale = Literal["slaney", "htk"]


def _hz_to_mel(freq: float, mel_scale: MelScale) -> float:
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: Tensor, mel_scale: MelScale) -> Tensor:
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * (logstep * (mels[log_t] - min_log_mel)).exp()
    return freqs


def _create_triangular_filterbank(
    all_freqs: Tensor,
    f_pts: Tensor,
) -> Tensor:
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    zero = Tensor.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = Tensor.maximum(zero, Tensor.minimum(down_slopes, up_slopes))
    return fb


def melscale_fbanks(
    n_mels: int,
    f_min: float,
    f_max: float,
    sample_rate: int,
    norm: Optional[str] = None,
    n_freqs=201,
    mel_scale: MelScale = "htk",
) -> Tensor:
    all_freqs = Tensor.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = Tensor.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    fb = _create_triangular_filterbank(all_freqs, f_pts)
    if norm is not None and norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    return fb


def melscale(
    x: Tensor, n_mels: int, f_min: float, f_max: float, sample_rate: int, n_fft: int
):
    fb = melscale_fbanks(
        n_mels,
        f_min,
        f_max,
        sample_rate,
        "htk",
        n_fft // 2 + 1,
    )
    return (x.transpose(-1, -2) @ fb).transpose(-1, -2)


def melspectrogram(
    x: Tensor,
    normalized: bool,
    power: float,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    sample_rate: int,
) -> Tensor:
    x = spectrogram(x, normalized, power, n_fft, hop_length)
    x = melscale(x, n_mels, f_min, f_max, sample_rate, n_fft)
    return x
