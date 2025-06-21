from tinygrad import Tensor


class WeightNormConvTranspose1d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight_v = Tensor.kaiming_uniform(
            in_channels, out_channels // groups, kernel_size
        )
        self.weight_g = (
            (self.weight_v.T.pow(2).sum(axis=(0, 2), keepdim=False).sqrt().clone())
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        if bias:
            self.bias = Tensor.zeros(out_channels)
        else:
            self.bias = None

    def weight(self):
        norm_v = self.weight_v.pow(2).sum(axis=(1, 2), keepdim=True).sqrt() + 1e-12
        g_reshaped = self.weight_g.reshape(self.in_channels, 1, 1)
        return g_reshaped * (self.weight_v / norm_v)

    def __call__(self, x: Tensor) -> Tensor:
        return x.conv_transpose2d(
            weight=self.weight(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
