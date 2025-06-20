import math
from tinygrad.tensor import Tensor


def _complex_mul(a: Tensor, b: Tensor) -> Tensor:
    a_real, a_imag = a[..., 0], a[..., 1]
    b_real, b_imag = b[..., 0], b[..., 1]
    real_part = a_real * b_real - a_imag * b_imag
    imag_part = a_real * b_imag + a_imag * b_real
    return Tensor.stack(real_part, imag_part, dim=-1)


def _bit_reverse_indices(n: int) -> list[int]:
    m = n.bit_length() - 1
    indices = list(range(n))
    for i in range(n):
        b = f"{i:0{m}b}"
        b_rev = b[::-1]
        indices[i] = int(b_rev, 2)
    return indices


def fft_in(x: Tensor) -> Tensor:
    N = x.shape[0]
    if (N & (N - 1) != 0) or N == 0:
        raise ValueError("FFT input size N must be a power of 2.")
    x_complex = Tensor.stack(x, Tensor.zeros(N), dim=1)
    indices = _bit_reverse_indices(N)
    x_reordered = x_complex[indices]
    log2_N = N.bit_length() - 1
    current_x = x_reordered
    for p in range(log2_N):
        size = 1 << (p + 1)
        half_size = size // 2
        k = Tensor.arange(half_size, requires_grad=False)
        angle = -2.0 * math.pi * k / size
        W_real = angle.cos()
        W_imag = angle.sin()
        W = Tensor.stack(W_real, W_imag, dim=1).reshape(1, half_size, 2)
        current_x_reshaped = current_x.reshape(N // size, size, 2)
        x_even = current_x_reshaped[:, :half_size, :]
        x_odd = current_x_reshaped[:, half_size:, :]
        term = _complex_mul(W, x_odd)
        res_top = x_even + term
        res_bottom = x_even - term
        current_x = Tensor.cat(res_top, res_bottom, dim=1).reshape(N, 2)

    return current_x


def fft(x: Tensor) -> Tensor:
    if x.ndim == 1:
        return fft_in(x)
    else:
        x = x.reshape(-1, x.shape[-1])
        result = Tensor.stack(*[fft_in(x_i) for x_i in x], dim=0)
        return result.reshape(*x.shape[:-1], -1, 2)
