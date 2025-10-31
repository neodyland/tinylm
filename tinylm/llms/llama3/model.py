from ..llama.model import (
    LlamaRMSNorm,
    LlamaBlock,
    llama_compute_attention_mask,
    llama_precompute_rope,
    llama_init_rope,
)
from ..llama.sample import llama_logits_sample
from ..llama.cache import LlamaAbstractKvCache
from ..abstract.causal_lm import (
    LlamaAbstractCausalLMForInference,
    LlamaAbstractCausalLMForTraining,
)
from tinygrad import nn, Tensor, TinyJit
from typing import List, Optional, Tuple


class Llama3RotaryEmbedding:
    def __init__(self, rope_theta: float, head_dim: int, ctx_len: int):
        inv_freq = llama_init_rope(rope_theta, head_dim)
        # TODO: I don't know how llama3's rope works.
        emb = llama_precompute_rope(inv_freq, ctx_len)
        self.cos = emb.cos()
        self.sin = emb.sin()

    def __call__(self, x: Tensor, pos_x: int, pos_y: int) -> Tuple[Tensor, Tensor]:
        return self.cos[:, pos_x:pos_y].cast(x.dtype), self.sin[:, pos_x:pos_y].cast(
            x.dtype
        )


class Llama3Model:
    def __init__(
        self,
        num_layers: int,
        dim: int,
        ffn_dim: int,
        kv_heads: int,
        head_dim: int,
        vocab_size: int,
        rope_theta: float,
        att_heads: int,
        ctx_len: int,
    ):
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.rotary_emb = Llama3RotaryEmbedding(rope_theta, head_dim, ctx_len)
        self.layers = [
            LlamaBlock(dim, kv_heads, head_dim, ffn_dim, att_heads)
            for _ in range(num_layers)
        ]
        self.norm = LlamaRMSNorm(dim)

    def __call__(
        self,
        x: Tensor,
        real_len: int,
        kv_caches: List[Optional[LlamaAbstractKvCache]],
    ) -> Tensor:
        x = self.embed_tokens(x)
        pos_x, pos_y = (real_len - x.shape[1], real_len)
        position_ids = Tensor.arange(pos_x, pos_y)
        attention_mask = llama_compute_attention_mask(
            x.dtype,
            x.shape[1],
            real_len,
            position_ids,
            x.shape[0],
        )
        position_embeddings = self.rotary_emb(x, pos_x, pos_y)
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(
                x, position_embeddings, attention_mask, real_len, kv_cache
            )
        x = self.norm(x)
        return x


class Llama3ModelForCasualLM(
    LlamaAbstractCausalLMForInference, LlamaAbstractCausalLMForTraining
):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        ffn_dim: int,
        kv_heads: int,
        head_dim: int,
        vocab_size: int,
        rope_theta: float,
        att_heads: int,
        ctx_len: int,
    ):
        self.ctx_len = ctx_len
        self.num_layers = num_layers
        self.model = Llama3Model(
            num_layers,
            dim,
            ffn_dim,
            kv_heads,
            head_dim,
            vocab_size,
            rope_theta,
            att_heads,
            ctx_len,
        )
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.kv_heads = kv_heads
        self.head_dim = head_dim

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        real_len = x.shape[1]
        x, _ = self.model(x, real_len, [None for _ in self.model.layers])
        x = self.lm_head(x)
        return x

    @TinyJit
    def inference(
        self,
        x: Tensor,
        real_len: int,
        kv_caches: List[Optional[LlamaAbstractKvCache]],
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tensor:
        x = self.model(x, real_len, kv_caches)
        x = self.lm_head(x[:, -1, :])
        return llama_logits_sample(x, temperature, top_p, top_k).realize()

    @TinyJit
    def prefill(
        self, x: Tensor, real_len: int, kv_caches: List[Optional[LlamaAbstractKvCache]]
    ) -> Tensor:
        x = self.model(x, real_len, kv_caches)
        return x.realize()
