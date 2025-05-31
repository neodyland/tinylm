from ..llama.model import (
    LlamaRMSNorm,
    LlamaSiluMLP,
    llama_apply_rotary_pos_emb,
    llama_attention,
    llama_compute_attention_mask,
    LlamaRotaryEmbedding,
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
        self.q_proj = nn.Linear(dim, att_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(att_heads * head_dim, dim, bias=False)
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
    ) -> Tuple[Tensor, Optional[LlamaAbstractKvCache]]:
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        query_states, key_states = llama_apply_rotary_pos_emb(
            query_states, key_states, position_embeddings[0], position_embeddings[1]
        )
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, real_len
            )

        attn_output = llama_attention(
            key_states,
            value_states,
            query_states,
            self.att_heads // self.kv_heads,
            self.scaling,
            attention_mask,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, kv_cache


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
    ) -> Tuple[Tensor, Optional[LlamaAbstractKvCache]]:
        residual = x
        x = self.input_layernorm(x)
        x, kv_cache = self.self_attn(
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
        return x, kv_cache


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
    ) -> Tuple[Tensor, List[Optional[LlamaAbstractKvCache]]]:
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
        updated_kv_caches = []
        for layer, kv_cache in zip(self.layers, kv_caches):
            x, kv_cache = layer(
                x, position_embeddings, attention_mask, real_len, kv_cache
            )
            updated_kv_caches.append(kv_cache)
        x = self.norm(x)
        return x, updated_kv_caches


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
    ) -> Tuple[Tensor, Tensor, List[Optional[LlamaAbstractKvCache]]]:
        x, kv_caches = self.model(x, real_len, kv_caches)
        x = self.lm_head(x[:, -1, :])
        return llama_logits_sample(x, temperature, top_p, top_k), x, kv_caches

    @TinyJit
    def prefill(
        self, x: Tensor, real_len: int, kv_caches: List[Optional[LlamaAbstractKvCache]]
    ) -> Tuple[Tensor, List[Optional[LlamaAbstractKvCache]]]:
        x, kv_caches = self.model(x, real_len, kv_caches)
        return x, kv_caches
