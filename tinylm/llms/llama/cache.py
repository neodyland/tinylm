from typing import Tuple
from tinygrad import Tensor
from tinygrad.tensor import DType
from ..abstract.cache import LlamaAbstractKvCache


class LlamaStaticKvCache(LlamaAbstractKvCache):
    def __init__(
        self, batch_size: int, kv_heads: int, ctx_len: int, head_dim: int, dtype: DType
    ):
        super().__init__(batch_size, kv_heads, ctx_len, head_dim)
        self.kv = (
            Tensor.zeros(
                2,
                batch_size,
                kv_heads,
                ctx_len,
                head_dim,
                dtype=dtype,
            )
            .contiguous()
            .realize()
        )

    def update(
        self, key: Tensor, value: Tensor, real_len: int
    ) -> Tuple[Tensor, Tensor]:
        self.kv[:, :, :, real_len - key.shape[2] : real_len, :].assign(Tensor.stack(key,value)).realize()
        key = self.kv[0, :, :, 0:real_len, :]
        value = self.kv[1, :, :, 0:real_len, :]
        return key, value
