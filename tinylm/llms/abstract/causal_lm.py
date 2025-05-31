from tinygrad import Tensor
from ..llama.cache import LlamaAbstractKvCache
from typing import List, Optional, Tuple


class LlamaAbstractCausalLMForInference:
    ctx_len: int
    num_layers: int

    def prefill(
        self, x: Tensor, real_len: int, kv_caches: List[Optional[LlamaAbstractKvCache]]
    ) -> Tuple[Tensor, List[Optional[LlamaAbstractKvCache]]]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def inference(
        self,
        x: Tensor,
        real_len: int,
        kv_caches: List[Optional[LlamaAbstractKvCache]],
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[Tensor, Tensor, List[Optional[LlamaAbstractKvCache]]]:
        raise NotImplementedError("This method should be implemented by subclasses.")


class LlamaAbstractCausalLMForTraining:
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")
