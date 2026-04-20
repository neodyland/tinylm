from ..llama.model import (
    LlamaRMSNorm,
    LlamaSiluMLP,
    llama_apply_rotary_pos_emb,
    llama_attention,
    llama_compute_attention_mask,
    LlamaRotaryEmbedding,
    LlamaNoBiasLinear,
)
from ..llama.sample import llama_logits_sample
from ..llama.cache import LlamaAbstractKvCache
from ..abstract.causal_lm import (
    LlamaAbstractCausalLMForInference,
    LlamaAbstractCausalLMForTraining,
)
from tinygrad import nn, Tensor, TinyJit
from typing import List, Optional, Tuple


class Qwen3Attention:
    def __init__(
        self,
        dim: int,
        kv_heads: int,
        head_dim: int,
        att_heads: int,
    ):
        # self.q_proj = LlamaNoBiasLinear(dim, att_heads * head_dim)
        # self.k_proj = LlamaNoBiasLinear(dim, kv_heads * head_dim)
        # self.v_proj = LlamaNoBiasLinear(dim, kv_heads * head_dim)
        self.qkv_proj = LlamaNoBiasLinear(dim, (att_heads + 2 * kv_heads) * head_dim)
        self.o_proj = LlamaNoBiasLinear(att_heads * head_dim, dim)
        self.q_norm = LlamaRMSNorm(head_dim)
        self.k_norm = LlamaRMSNorm(head_dim)
        self.att_heads = att_heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.scaling: int = head_dim**-0.5

    def __call__(
        self,
        x: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        real_len: int,
        kv_cache: Optional[LlamaAbstractKvCache],
    ) -> Tensor:
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q, k, v = self.qkv_proj(x).split(
            [
                self.att_heads * self.head_dim,
                self.kv_heads * self.head_dim,
                self.kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)
        q, k = llama_apply_rotary_pos_emb(
            q, k, position_embeddings[0], position_embeddings[1]
        )
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, real_len)

        attn_output = llama_attention(
            k,
            v,
            q,
            self.att_heads // self.kv_heads,
            self.scaling,
            attention_mask,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3Block:
    def __init__(
        self,
        dim: int,
        kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        att_heads: int,
    ):
        self.self_attn = Qwen3Attention(dim, kv_heads, head_dim, att_heads)
        self.mlp = LlamaSiluMLP(dim, ffn_dim)
        self.input_layernorm = LlamaRMSNorm(dim)
        self.post_attention_layernorm = LlamaRMSNorm(dim)

    def __call__(
        self,
        x: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        real_len: int,
        kv_cache: Optional[LlamaAbstractKvCache] = None,
    ) -> Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            position_embeddings,
            attention_mask,
            real_len,
            kv_cache,
        )
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3Model:
    def __init__(
        self,
        num_layers: int,
        dim: int,
        ffn_dim: int,
        kv_heads: int,
        head_dim: int,
        vocab_size: int,
        rope_theta: int,
        att_heads: int,
        ctx_len: int,
    ):
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.rotary_emb = LlamaRotaryEmbedding(rope_theta, head_dim, ctx_len)
        self.layers = [
            Qwen3Block(dim, kv_heads, head_dim, ffn_dim, att_heads)
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
        attention_mask = (
            llama_compute_attention_mask(
                x.dtype,
                x.shape[1],
                real_len,
                position_ids,
                x.shape[0],
            )
            if x.shape[1] > 1
            else None
        )
        position_embeddings = self.rotary_emb(x, pos_x, pos_y)
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, position_embeddings, attention_mask, real_len, kv_cache)
        x = self.norm(x)
        return x


class Qwen3ModelForCasualLM(
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
        rope_theta: int,
        att_heads: int,
        ctx_len: int,
    ):
        self.ctx_len = ctx_len
        self.num_layers = num_layers
        self.model = Qwen3Model(
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
        self.lm_head = LlamaNoBiasLinear(dim, vocab_size)
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
        return llama_logits_sample(x, temperature, top_p, top_k)

    @TinyJit
    def prefill(
        self, x: Tensor, real_len: int, kv_caches: List[Optional[LlamaAbstractKvCache]]
    ) -> Tensor:
        x = self.model(x, real_len, kv_caches)
        return x
