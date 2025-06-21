from .pack import pack, unpack
from tinygrad import Tensor
import numpy as np
import pytest
import torch
from ..testutil.torch_close import TorchTinygradCloseTest
import einops


class PackTest(TorchTinygradCloseTest):
    def __init__(
        self,
        pattern: str,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        super().__init__(atol, rtol)
        self.pattern = pattern

    def tinygrad(self, data: Tensor) -> Tensor:
        packed_tensor, packed_shapes = pack([data], self.pattern)
        unpacked_tensor = unpack(packed_tensor, packed_shapes, self.pattern)[0]
        return unpacked_tensor

    def torch(self, data: torch.Tensor) -> torch.Tensor:
        packed_tensor, packed_shapes = einops.pack([data], self.pattern)
        unpacked_tensor = einops.unpack(packed_tensor, packed_shapes, self.pattern)[0]
        return unpacked_tensor


@pytest.mark.parametrize(
    "pattern, shape",
    [
        ("b * d", (2, 3, 4)),
        ("* b d", (2, 3, 4)),
        ("b d *", (2, 3, 4)),
        ("b * c d", (2, 3, 4, 5)),
        ("* b c d", (2, 3, 4, 5)),
        ("b c * d", (2, 3, 4, 5)),
        ("b c d *", (2, 3, 4, 5)),
        ("* b c d e", (2, 3, 4, 5, 6)),
        ("b * c d e", (2, 3, 4, 5, 6)),
        ("b c * d e", (2, 3, 4, 5, 6)),
        ("b c d * e", (2, 3, 4, 5, 6)),
    ],
)
def test_pack_unpack(pattern, shape):
    data = np.random.randn(*shape).astype(np.float32)
    test = PackTest(pattern)
    test.run(data)
