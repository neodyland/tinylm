from tinygrad import Tensor, nn, dtypes, TinyJit, Variable, Context
from tinygrad.dtype import DType
from transformers import AutoTokenizer
from typing import Tuple, Optional, List, Dict, Generator, Union
from huggingface_hub import hf_hub_download
import time
from dataclasses import dataclass
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live


class Qwen3RMSNorm:
    def __init__(self, dim: int, eps=1e-6):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.cast(dtypes.float)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * (variance + self.eps).rsqrt()
        return self.weight * x.cast(input_dtype)


class Qwen3MLP:
    def __init__(self, dim: int, ffn_dim: int):
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))
        return down_proj


def rotate_half(x: Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return Tensor.cat(-x2, x1, dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3Attention:
    def __init__(
        self,
        dim: int,
        kv_heads: int,
        head_dim: int,
        att_heads: int,
        ctx_len: int,
    ):
        self.q_proj = nn.Linear(dim, att_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(att_heads * head_dim, dim, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim)
        self.k_norm = Qwen3RMSNorm(head_dim)
        self.att_heads = att_heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.scaling: int = head_dim**-0.5
        self.k_cache = (
            Tensor.zeros(
                1,
                self.kv_heads,
                ctx_len,
                self.head_dim,
            )
            .contiguous()
            .realize()
        )
        self.v_cache = (
            Tensor.zeros(
                1,
                self.kv_heads,
                ctx_len,
                self.head_dim,
            )
            .contiguous()
            .realize()
        )

    def __call__(
        self,
        x: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        real_len: int,
    ) -> Tensor:
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, *position_embeddings
        )
        self.k_cache[:, :, real_len - x.shape[1] : real_len, :].assign(
            key_states
        ).realize()
        self.v_cache[:, :, real_len - x.shape[1] : real_len, :].assign(
            value_states
        ).realize()
        key_states = self.k_cache[:, :, 0:real_len, :]
        value_states = self.v_cache[:, :, 0:real_len, :]

        key_states = repeat_kv(key_states, self.att_heads // self.kv_heads)
        value_states = repeat_kv(value_states, self.att_heads // self.kv_heads)
        attn_weights = (query_states @ key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.softmax(axis=-1, dtype=dtypes.float).cast(
            query_states.dtype
        )
        attn_output = attn_weights @ value_states
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
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
        ctx_len: int,
    ):
        self.self_attn = Qwen3Attention(dim, kv_heads, head_dim, att_heads, ctx_len)
        self.mlp = Qwen3MLP(dim, ffn_dim)
        self.input_layernorm = Qwen3RMSNorm(dim)
        self.post_attention_layernorm = Qwen3RMSNorm(dim)

    def __call__(
        self,
        x: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        real_len: int,
    ) -> Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            position_embeddings,
            attention_mask,
            real_len,
        )
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3RotaryEmbedding:
    def __init__(self, rope_theta: int, head_dim: int, ctx_len: int):
        inv_freq = 1.0 / (
            rope_theta
            ** (
                Tensor.arange(0, head_dim, 2, dtype=dtypes.int64).cast(dtypes.float)
                / head_dim
            )
        )
        position_ids = (
            Tensor.arange(ctx_len, dtype=dtypes.int64).cast(dtypes.float).unsqueeze(0)
        )
        inv_freq_expanded = (
            inv_freq[None, :, None]
            .cast(dtypes.float)
            .expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].cast(dtypes.float)
        freqs = (
            inv_freq_expanded.cast(dtypes.float)
            @ position_ids_expanded.cast(dtypes.float)
        ).transpose(1, 2)
        emb = Tensor.cat(freqs, freqs, dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()

    def __call__(self, x: Tensor, pos_x: int, pos_y: int) -> Tuple[Tensor, Tensor]:
        return self.cos[:, pos_x:pos_y].cast(x.dtype), self.sin[:, pos_x:pos_y].cast(
            x.dtype
        )


def compute_attention_mask(
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
        self.rotary_emb = Qwen3RotaryEmbedding(rope_theta, head_dim, ctx_len)
        self.layers = [
            Qwen3Block(dim, kv_heads, head_dim, ffn_dim, att_heads, ctx_len)
            for _ in range(num_layers)
        ]
        self.norm = Qwen3RMSNorm(dim)

    def __call__(
        self,
        x: Tensor,
        real_len: int,
    ) -> Tensor:
        x = self.embed_tokens(x)
        pos_x, pos_y = (real_len - x.shape[1], real_len)
        position_ids = Tensor.arange(pos_x, pos_y)
        attention_mask = compute_attention_mask(
            x.dtype,
            x.shape[1],
            real_len,
            position_ids,
            x.shape[0],
        )
        position_embeddings = self.rotary_emb(x, pos_x, pos_y)
        for layer in self.layers:
            x = layer(x, position_embeddings, attention_mask, real_len)
        x = self.norm(x)
        return x


class Qwen3ModelForCasualLM:
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
        self.len_var = Variable(
            "real_len",
            1,
            ctx_len,
        )
        self.ctx_len = ctx_len

    def __call__(
        self,
        x: Tensor,
        real_len: int,
    ) -> Tensor:
        x = self.model(x, real_len)
        x = self.lm_head(x[:, -1, :])
        return x

    @TinyJit
    @Context(BEAM=2)
    def fused_inference(self, tensor: Tensor, real_len: int) -> Tensor:
        logits = self(tensor, real_len)
        return sample(logits)

    @TinyJit
    @Context(BEAM=2)
    def fused_prefill(self, tensor: Tensor, real_len: int) -> Tensor:
        logits = self(tensor, real_len)
        return logits


def sample(
    logits: Tensor,
    temperature=0.6,
    top_p=0.8,
    top_k=20,
) -> Tensor:
    if temperature < 1e-6:
        return logits.argmax().unsqueeze(0)
    logits = logits.flatten()

    logits = (logits != logits).where(-float("inf"), logits)

    t = (logits / temperature).softmax()

    counter, counter2 = (
        Tensor.arange(t.numel(), device=logits.device).contiguous(),
        Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous(),
    )

    if top_k:
        output, output_indices = (
            Tensor.zeros(top_k, device=logits.device).contiguous(),
            Tensor.zeros(top_k, device=logits.device, dtype=dtypes.int32).contiguous(),
        )
        for i in range(top_k):
            t_argmax = (
                t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1
            ).cast(dtypes.default_int)
            output = output + t_max.unsqueeze(0).pad(((i, top_k - i - 1),))
            output_indices = output_indices + t_argmax.unsqueeze(0).pad(
                ((i, top_k - i - 1),)
            )
            t = (counter == t_argmax).where(0, t)

        output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
        output = (output_cumsum >= (1 - top_p)) * output
        output_indices = (output_cumsum >= (1 - top_p)) * output_indices

        output_idx = output.multinomial()
        output_token = output_indices[output_idx]
    else:
        output_token = t.multinomial()

    return output_token


@dataclass
class GenerationFinal:
    input_tokens: int
    output_tokens: int
    prefill_time: float
    generate_time: float
    sequence: str


def generate(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    model: Qwen3ModelForCasualLM,
    tokenizer: AutoTokenizer,
    thinking: bool,
    prefill_chunk_size: int,
) -> Generator[Union[str, GenerationFinal], None, None]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    input_ids = tokenizer(text, return_tensors="np").input_ids.tolist()[0]
    if len(input_ids) + max_new_tokens > model.ctx_len:
        raise ValueError(
            f"Input length ({len(input_ids)}) + max_new_tokens ({max_new_tokens}) exceeds context length ({model.ctx_len})."
        )
    prefill_time = time.time()
    for i in range(0, len(input_ids), prefill_chunk_size):
        chunk_end = min(i + prefill_chunk_size, len(input_ids))
        chunk = input_ids[i:chunk_end]
        if len(chunk) < prefill_chunk_size:
            chunk += [tokenizer.pad_token_id] * (prefill_chunk_size - len(chunk))
        tensor = Tensor(chunk, dtype=dtypes.int64).unsqueeze(0)
        model.fused_prefill(
            tensor,
            model.len_var.bind(i + prefill_chunk_size),
        ).realize()
    prefill_time = time.time() - prefill_time
    input_len = len(input_ids)
    generate_time = time.time()
    for _ in range(max_new_tokens):
        tensor = Tensor([input_ids[-1]], dtype=dtypes.int64).unsqueeze(0)
        next_token = (
            model.fused_inference(
                tensor,
                model.len_var.bind(len(input_ids)),
            )
            .numpy()[0]
            .item()
        )
        if next_token == tokenizer.eos_token_id:
            break
        input_ids.append(next_token)
        yield (tokenizer.decode(input_ids[input_len:]))
    generate_time = time.time() - generate_time
    yield GenerationFinal(
        input_tokens=input_len,
        output_tokens=len(input_ids) - input_len,
        prefill_time=prefill_time,
        generate_time=generate_time,
        sequence=tokenizer.decode(input_ids).split("</think>")[-1].strip(),
    )


def styled_markdown(text: str) -> Markdown:
    text = text.replace("<think>\n\n</think>", "")
    text = text.replace("\n", "\n\n")
    if "</think>" in text:
        thinking_content = text.split("</think>")[0].split("<think>")[-1].strip()
        answer_content = text.split("</think>")[-1].strip()
        thinking_content = f"> {thinking_content.replace('\n\n', '\n>\n> ')}"
        text = f"{thinking_content}\n\n{answer_content}"
    return Markdown(
        text,
    )


def main():
    Tensor.no_grad = True
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-0.6B")
    model = Qwen3ModelForCasualLM(
        num_layers=28,
        dim=1024,
        ffn_dim=3072,
        kv_heads=8,
        head_dim=128,
        vocab_size=151936,
        rope_theta=1000000,
        att_heads=16,
        ctx_len=40960,
    )
    prefill_chunk_size = 256
    path = hf_hub_download("unsloth/Qwen3-0.6B", "model.safetensors")
    state_dict = nn.state.safe_load(path)
    state_dict["model.rotary_emb.sin"] = model.model.rotary_emb.sin
    state_dict["model.rotary_emb.cos"] = model.model.rotary_emb.cos
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    for layer in range(len(model.model.layers)):
        state_dict[f"model.layers.{layer}.self_attn.k_cache"] = model.model.layers[
            layer
        ].self_attn.k_cache.cast(state_dict["lm_head.weight"].dtype)
        state_dict[f"model.layers.{layer}.self_attn.v_cache"] = model.model.layers[
            layer
        ].self_attn.v_cache.cast(state_dict["lm_head.weight"].dtype)
    nn.state.load_state_dict(model, state_dict)
    total_params = sum(param.numel() for param in nn.state.get_parameters(model))
    console = Console()
    console.print(
        f"Total parameters in the model: {total_params / 1e6:.2f}M", style="green"
    )
    console.print("Start warmup...", style="green")
    model.fused_prefill(
        Tensor([0] * prefill_chunk_size, dtype=dtypes.int64).unsqueeze(0),
        model.len_var.bind(prefill_chunk_size),
    ).realize()
    model.fused_inference(
        Tensor([0], dtype=dtypes.int64).unsqueeze(0),
        model.len_var.bind(model.ctx_len // 2),
    ).realize()
    console.print("Warmup done.", style="green")
    chat = []
    while True:
        console.print("User: ", style="cyan")
        user_input = console.input().strip()
        if user_input == "exit":
            break
        if user_input == "clear":
            chat.clear()
            console.print("Chat history cleared.", style="green")
            continue
        chat.append({"role": "user", "content": user_input})
        console.print("Assistant: ", style="cyan")
        console.print()
        live = Live(console=console)
        live.start()
        for chunk in generate(
            chat,
            max_new_tokens=model.ctx_len // 2,
            model=model,
            tokenizer=tokenizer,
            thinking=True,
            prefill_chunk_size=prefill_chunk_size,
        ):
            if isinstance(chunk, str):
                live.update(styled_markdown(chunk), refresh=True)
            else:
                live.stop()
                console.print(
                    f"Input tokens: {chunk.input_tokens}\nOutput tokens: {chunk.output_tokens}\nPrefill TPS: {chunk.input_tokens / chunk.prefill_time:.2f}\nGenerate TPS: {chunk.output_tokens / chunk.generate_time:.2f}",
                    style="green",
                )
                chat.append(
                    {
                        "role": "assistant",
                        "content": chunk.sequence,
                    }
                )


if __name__ == "__main__":
    main()
