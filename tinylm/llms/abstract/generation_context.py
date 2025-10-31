from tinygrad.tensor import DType
from tinygrad import Variable, dtypes, Tensor
from typing import List, Generator, Union, Literal, Optional
from dataclasses import dataclass
from .causal_lm import LlamaAbstractCausalLMForInference
from .cache import LlamaAbstractKvCache


@dataclass
class LlamaGenerationChunkFinal:
    type: Literal["end"] = "end"
    reason: Literal["eos", "max_new_tokens"] = "eos"


@dataclass
class LlamaGenerationChunkToken:
    token: int
    type: Literal["token"] = "token"


@dataclass
class LlamaGenerationChunkPrefillStart:
    prefill_tokens: int
    type: Literal["prefill_start"] = "prefill_start"


@dataclass
class LlamaGenerationChunkPrefillEnd:
    type: Literal["prefill_end"] = "prefill_end"


type LlamaGenerationChunk = Union[
    LlamaGenerationChunkFinal,
    LlamaGenerationChunkToken,
    LlamaGenerationChunkPrefillStart,
    LlamaGenerationChunkPrefillEnd,
]


class LlamaAbstractGenerationContext:
    kv_caches: Optional[List[LlamaAbstractKvCache]]
    model: LlamaAbstractCausalLMForInference

    def reset_kv_caches(self, batch_size: int, dtype: DType):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __init__(
        self,
        model: LlamaAbstractCausalLMForInference,
        batch_size: int,
        prefill_chunk_size: int,
        dtype: DType,
        pad_token_id: int,
        eos_token_id: Union[int, List[int]],
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        self.model = model
        self.reset_kv_caches(batch_size, dtype)
        self.prefill_chunk_size = prefill_chunk_size
        self.len_var = Variable(
            "real_len",
            1,
            model.ctx_len,
        )
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def warmup(self):
        prefill_times = 4  # just a magic number for now
        for _ in range(prefill_times):
            _ = self.model.prefill(
                Tensor([0] * self.prefill_chunk_size, dtype=dtypes.int64).unsqueeze(0),
                self.len_var.bind(self.prefill_chunk_size),
                self.kv_caches,
            )
            _ = self.model.inference(
                Tensor([0], dtype=dtypes.int64).unsqueeze(0),
                self.len_var.bind(self.model.ctx_len // 2),
                self.kv_caches,
                self.temperature,
                self.top_p,
                self.top_k,
            )

    def is_eos(self, token: int) -> bool:
        if isinstance(self.eos_token_id, list):
            return token in self.eos_token_id
        return token == self.eos_token_id

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int,
    ) -> Generator[LlamaGenerationChunk, None, None]:
        if len(input_ids) + max_new_tokens > self.model.ctx_len:
            raise ValueError(
                f"Input length ({len(input_ids)}) + max_new_tokens ({max_new_tokens}) exceeds context length ({self.model.ctx_len})."
            )
        yield LlamaGenerationChunkPrefillStart(
            prefill_tokens=(len(input_ids) // self.prefill_chunk_size + 1)
            * self.prefill_chunk_size
        )
        for i in range(0, len(input_ids), self.prefill_chunk_size):
            chunk_end = min(i + self.prefill_chunk_size, len(input_ids))
            chunk = input_ids[i:chunk_end]
            if len(chunk) < self.prefill_chunk_size:
                chunk += [self.pad_token_id] * (self.prefill_chunk_size - len(chunk))
            tensor = Tensor(chunk, dtype=dtypes.int64, requires_grad=False).unsqueeze(0)
            _ = self.model.prefill(
                tensor,
                self.len_var.bind(i + self.prefill_chunk_size),
                self.kv_caches,
            )
        yield LlamaGenerationChunkPrefillEnd()
        reason = "max_new_tokens"
        for _ in range(max_new_tokens):
            tensor = Tensor(
                [input_ids[-1]], dtype=dtypes.int64, requires_grad=False
            ).unsqueeze(0)
            x = self.model.inference(
                tensor,
                self.len_var.bind(len(input_ids)),
                self.kv_caches,
                self.temperature,
                self.top_p,
                self.top_k,
            )
            next_token = x[0].item()
            if self.is_eos(next_token):
                reason = "eos"
                break
            input_ids.append(next_token)
            yield LlamaGenerationChunkToken(
                token=next_token,
            )
        yield LlamaGenerationChunkFinal(reason=reason)
