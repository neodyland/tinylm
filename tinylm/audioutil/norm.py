from tinygrad import Tensor
from typing import Union, Optional, List, Tuple

type NormP = Union[int, float, str]
type DimType = Optional[Union[int, List[int], Tuple[int, ...]]]


def norm(
    x: Tensor,
    p: NormP = "fro",
    dim: DimType = None,
    keepdim: bool = False,
) -> Tensor:
    axis = dim
    if p == "fro":
        p = 2
    if p == float("inf"):
        return x.abs().max(axis=axis, keepdim=keepdim)
    if p == float("-inf"):
        return x.abs().min(axis=axis, keepdim=keepdim)
    if p == 1:
        return x.abs().sum(axis=axis, keepdim=keepdim)
    if p == 2:
        return x.pow(2).sum(axis=axis, keepdim=keepdim).sqrt()
    if not isinstance(p, (int, float)) or p <= 0:
        raise ValueError(
            f"Unsupported norm order p={p}. Only positive numbers, 'fro', 'inf', and '-inf' are supported."
        )
    return x.abs().pow(p).sum(axis=axis, keepdim=keepdim).pow(1.0 / p)
