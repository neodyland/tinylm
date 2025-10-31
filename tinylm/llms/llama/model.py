from tinygrad import Tensor, dtypes, nn, TinyJit
from tinygrad.tensor import DType
from typing import Tuple, Optional, List
from .cache import LlamaAbstractKvCache
from .sample import llama_logits_sample
from ..abstract.causal_lm import (
    LlamaAbstractCausalLMForInference,
    LlamaAbstractCausalLMForTraining,
)


class LlamaRMSNorm:
    def __init__(self, dim: int, eps=1e-6):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.cast(dtypes.float)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * (variance + self.eps).rsqrt()
        return self.weight * x.cast(input_dtype)


class LlamaSiluMLP:
    def __init__(self, dim: int, ffn_dim: int):
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))
        return down_proj


def llama_rotate_half(x: Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return Tensor.cat(-x2, x1, dim=-1)


def llama_apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (llama_rotate_half(q) * sin)
    k_embed = (k * cos) + (llama_rotate_half(k) * sin)
    return q_embed, k_embed


def llama_init_rope(rope_theta: float, head_dim: int):
    inv_freq = 1.0 / (
        rope_theta
        ** (
            Tensor.arange(0, head_dim, 2, dtype=dtypes.int64).cast(dtypes.float)
            / head_dim
        )
    )
    return inv_freq


def llama_precompute_rope(inv_freq: Tensor, ctx_len: int):
    position_ids = (
        Tensor.arange(ctx_len, dtype=dtypes.int64).cast(dtypes.float).unsqueeze(0)
    )
    inv_freq_expanded = (
        inv_freq[None, :, None].cast(dtypes.float).expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].cast(dtypes.float)
    freqs = (
        inv_freq_expanded.cast(dtypes.float) @ position_ids_expanded.cast(dtypes.float)
    ).transpose(1, 2)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    return emb


class LlamaRotaryEmbedding:
    def __init__(self, rope_theta: int, head_dim: int, ctx_len: int):
        inv_freq = llama_init_rope(rope_theta, head_dim)
        emb = llama_precompute_rope(inv_freq, ctx_len)
        self.cos = emb.cos()
        self.sin = emb.sin()

    def __call__(self, x: Tensor, pos_x: int, pos_y: int) -> Tuple[Tensor, Tensor]:
        return self.cos[:, pos_x:pos_y].cast(x.dtype), self.sin[:, pos_x:pos_y].cast(
            x.dtype
        )


def llama_repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def llama_attention(
    key_states: Tensor,
    value_states: Tensor,
    query_states: Tensor,
    n_rep: int,
    scaling: float,
    attention_mask: Optional[Tensor],
) -> Tensor:
    key_states = llama_repeat_kv(key_states, n_rep)
    value_states = llama_repeat_kv(value_states, n_rep)
    attn_weights = (query_states @ key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = attn_weights.softmax(axis=-1, dtype=dtypes.float).cast(
        query_states.dtype
    )
    attn_output = attn_weights @ value_states
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def llama_compute_attention_mask(
    dtype: DType,
    source_length: int,
    target_length: int,
    position_ids: Tensor,
    batch_size: int,
) -> Tensor:
    causal_mask = Tensor.full(
        (source_length, target_length), fill_value=-100, dtype=dtype
    )
    diagonal_attend_mask = Tensor.arange(0, target_length) > position_ids.reshape(-1, 1)
    causal_mask *= diagonal_attend_mask
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return causal_mask


class LlamaAttention:
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

        query_states = self.q_proj(x).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(x).view(hidden_shape).transpose(1, 2)
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
        return attn_output


class LlamaBlock:
    def __init__(
        self,
        dim: int,
        kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        att_heads: int,
    ):
        self.self_attn = LlamaAttention(dim, kv_heads, head_dim, att_heads)
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


class LlamaModel:
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


class LlamaModelForCasualLM(
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
        self.model = LlamaModel(
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
