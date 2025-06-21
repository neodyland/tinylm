import torch
from ..testutil.torch_close import TorchTinygradCloseTest
import pytest
import numpy as np
from typing import Optional
from tinygrad import Tensor
from .norm import norm, NormP, DimType


class NormTest(TorchTinygradCloseTest):
    def __init__(
        self,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        p: NormP = "fro",
        dim: DimType = None,
        keepdim: bool = False,
    ):
        super().__init__(atol, rtol)
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def tinygrad(self, data: Tensor) -> Tensor:
        return norm(data, self.p, self.dim, self.keepdim)

    def torch(self, data: torch.Tensor) -> torch.Tensor:
        return torch.norm(data, p=self.p, dim=self.dim, keepdim=self.keepdim)


@pytest.mark.parametrize("p", ["fro", 1.0, float("inf")])
def test_norm_1d(p: Optional[NormP]):
    data = np.random.randn(10).astype(np.float32)
    test = NormTest(p=p)
    test.run(data)


@pytest.mark.parametrize(
    "p, dim, keepdim",
    [
        ("fro", None, False),
        (2.0, None, False),
        (1.0, None, False),
        (3.0, None, False),
        (2.0, 0, False),
        (1.0, 1, False),
        (float("inf"), 1, False),
        (2.0, 1, True),
        (1.0, 0, True),
    ],
)
def test_norm_2d(p: NormP, dim: Optional[DimType], keepdim: bool):
    data = np.random.randn(3, 4).astype(np.float32)
    test = NormTest(p=p, dim=dim, keepdim=keepdim)
    test.run(data)


@pytest.mark.parametrize(
    "p, dim",
    [
        (2.0, 1),
        (1.0, (0, 2)),
    ],
)
def test_norm_3d(p: NormP, dim: DimType):
    data = np.arange(-12, 12, dtype=np.float32).reshape(2, 3, 4)
    test = NormTest(p=p, dim=dim)
    test.run(data)
