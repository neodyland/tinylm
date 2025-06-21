import torch
from ..testutil.torch_close import TorchTinygradCloseTest
import pytest
import numpy as np
from tinygrad import Tensor
from .wnconv1d import WeightNormConv1d


class WnConv1dTest(TorchTinygradCloseTest):
    def __init__(
        self,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        in_channels: int = 3,
        out_channels: int = 8,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        bias: bool = True,
    ):
        super().__init__(atol, rtol)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.tg_mod = WeightNormConv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        self.torch_mod = torch.nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        self.torch_mod = torch.nn.utils.weight_norm(
            self.torch_mod, name="weight", dim=0
        )
        self.tg_mod.weight_v.assign(self.torch_mod.weight_v.detach().numpy())
        self.tg_mod.weight_g.assign(
            self.torch_mod.weight_g.detach().numpy().reshape(-1, 1, 1)
        )
        if self.bias:
            self.tg_mod.bias.assign(self.torch_mod.bias.detach().numpy())

    def tinygrad(self, data: Tensor) -> Tensor:
        return self.tg_mod(data)

    def torch(self, data: torch.Tensor) -> torch.Tensor:
        return self.torch_mod(data)


@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("in_channels", [3, 5])
@pytest.mark.parametrize("seq_len", [10, 20])
def test_wnconv1d(bs, in_channels, seq_len):
    inputs = np.random.randn(bs, in_channels, seq_len).astype(np.float32)
    test = WnConv1dTest(
        in_channels=in_channels,
    )
    test.run(inputs)
