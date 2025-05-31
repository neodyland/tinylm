from ..abstract.generation_context import LlamaAbstractGenerationContext
from tinygrad.tensor import DType
from .cache import LlamaStaticKvCache


class LlamaGenerationContext(LlamaAbstractGenerationContext):
    def reset_kv_caches(self, batch_size: int, dtype: DType):
        self.kv_caches = [
            LlamaStaticKvCache(
                batch_size,
                self.model.kv_heads,
                self.model.ctx_len,
                self.model.head_dim,
                dtype,
            )
            for _ in range(self.model.num_layers)
        ]
