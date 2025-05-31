from ..abstract.generation_context import LlamaAbstractGenerationContext
from tinygrad.tensor import DType
from .cache import LlamaStaticKvCache
from ..abstract.causal_lm import LlamaAbstractCausalLMForInference


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

    def __init__(
        self,
        model: LlamaAbstractCausalLMForInference,
        batch_size: int,
        prefill_chunk_size: int,
        dtype: DType,
        pad_token_id: int,
        eos_token_id: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        super().__init__(
            model,
            batch_size,
            prefill_chunk_size,
            dtype,
            pad_token_id,
            eos_token_id,
            temperature,
            top_p,
            top_k,
        )
