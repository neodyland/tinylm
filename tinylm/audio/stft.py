from tinygrad import Tensor
from .fft import fft


def stft(
    y: Tensor,
    n_fft: int,
    hop_length: int,
    window: Tensor,
    center: bool = True,
) -> Tensor:
    if y.ndim != 1:
        raise ValueError("Input signal y must be a 1D tensor.")
    if len(window) != n_fft:
        raise ValueError("Window length must be equal to n_fft.")
    if center:
        pad_amount = n_fft // 2
        y_padded = y.pad(((pad_amount, pad_amount),), mode="reflect")
    else:
        y_padded = y
    num_samples = y_padded.shape[0]
    num_frames = 1 + (num_samples - n_fft) // hop_length
    if num_frames < 0:
        num_frames = 0
    stft_matrix_list = []
    for i in range(num_frames):
        start_index = i * hop_length
        end_index = start_index + n_fft
        frame = y_padded[start_index:end_index]
        windowed_frame = frame * window
        fft_result = fft(windowed_frame.reshape(1, n_fft))[0]
        stft_matrix_list.append(fft_result)
    if not stft_matrix_list:
        return Tensor.empty((1 + n_fft // 2, 0), dtype=y.dtype)
    stft_matrix = Tensor.stack(*stft_matrix_list, dim=-1)
    stft_matrix = stft_matrix[: 1 + n_fft // 2, :]
    return stft_matrix
