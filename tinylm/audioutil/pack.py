from tinygrad.tensor import Tensor
from typing import List, Tuple

PackedShapes = List[List[int]]


def pack(tensors: List[Tensor], pattern: str) -> Tuple[Tensor, PackedShapes]:
    if not all(isinstance(t, Tensor) for t in tensors):
        raise TypeError("All inputs must be tinygrad.Tensors")
    if not tensors:
        raise ValueError("Input tensor list cannot be empty")
    if pattern.count("*") != 1:
        raise ValueError("Pattern for pack must contain exactly one '*'")
    dims = pattern.split()
    try:
        pack_axis = dims.index("*")
    except ValueError:
        raise ValueError(
            f"Invalid pattern '{pattern}', '*' not found in split dimensions."
        )
    if any(t.ndim != len(dims) for t in tensors):
        raise ValueError(
            f"Pattern '{pattern}' has {len(dims)} dims, but tensors have different numbers of dimensions."
        )
    packed_shapes: PackedShapes = [[t.shape[pack_axis]] for t in tensors]
    packed_tensor = Tensor.cat(*tensors, dim=pack_axis)
    return packed_tensor, packed_shapes


def unpack(tensor: Tensor, ps: PackedShapes, pattern: str) -> List[Tensor]:
    if not isinstance(tensor, Tensor):
        raise TypeError("Input must be a tinygrad.Tensor")
    if pattern.count("*") != 1:
        raise ValueError("Pattern for unpack must contain exactly one '*'")
    dims = pattern.split()
    try:
        unpack_axis = dims.index("*")
    except ValueError:
        raise ValueError(
            f"Invalid pattern '{pattern}', '*' not found in split dimensions."
        )
    if tensor.ndim != len(dims):
        raise ValueError(
            f"Pattern '{pattern}' has {len(dims)} dims, but tensor has {tensor.ndim} dimensions."
        )
    split_sizes = [s[0] for s in ps]

    if sum(split_sizes) != tensor.shape[unpack_axis]:
        raise ValueError(
            f"Sum of packed shapes {sum(split_sizes)} does not match tensor "
            f"dimension {tensor.shape[unpack_axis]} at axis {unpack_axis}"
        )
    unpacked_tensors = []
    current_offset = 0
    for size in split_sizes:
        slicing = [slice(None)] * tensor.ndim
        slicing[unpack_axis] = slice(current_offset, current_offset + size)
        unpacked_chunk = tensor[tuple(slicing)]
        unpacked_tensors.append(unpacked_chunk)
        current_offset += size
    return unpacked_tensors
