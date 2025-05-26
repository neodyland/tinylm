from typing import Tuple
from tinygrad import Tensor
from tinygrad.tensor import DType
from ..abstract.cache import LlamaAbstractKvCache


class LlamaStaticKvCache(LlamaAbstractKvCache):
    def __init__(
        self, batch_size: int, kv_heads: int, ctx_len: int, head_dim: int, dtype: DType
    ):
        super().__init__(batch_size, kv_heads, ctx_len, head_dim)
        self.key = (
            Tensor.zeros(
                batch_size,
                kv_heads,
                ctx_len,
                head_dim,
                dtype=dtype,
            )
            .contiguous()
            .realize()
        )
        self.value = (
            Tensor.zeros(
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
        self.key[:, :, real_len - key.shape[2] : real_len, :].assign(key).realize()
        self.value[:, :, real_len - key.shape[2] : real_len, :].assign(value).realize()
        key = self.key[:, :, 0:real_len, :]
        value = self.value[:, :, 0:real_len, :]
        return key, value
