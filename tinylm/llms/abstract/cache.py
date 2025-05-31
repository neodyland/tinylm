from typing import Tuple
from tinygrad import Tensor


class LlamaAbstractKvCache:
    def __init__(self, batch_size: int, kv_heads: int, ctx_len: int, head_dim: int):
        self.batch_size = batch_size
        self.kv_heads = kv_heads
        self.ctx_len = ctx_len
        self.head_dim = head_dim

    def update(
        self, key: Tensor, value: Tensor, real_len: int
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("This method should be implemented by subclasses.")
