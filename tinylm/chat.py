from tinygrad import Tensor, nn, Device
from tinygrad.tensor import DType
from transformers import AutoTokenizer
from typing import Dict, Tuple
from huggingface_hub import hf_hub_download
import time
from rich.console import Console
from rich.live import Live
from tinylm.llms.llama3.model import Llama3ModelForCasualLM
from tinylm.llms.llama.model import LlamaModelForCasualLM
from tinylm.llms.qwen3.model import Qwen3ModelForCasualLM
from tinylm.llms.llama.generation_context import LlamaGenerationContext
from tinylm.clidefs import ModelLiteral
from tinylm.remote_chat import styled_markdown
from tinylm.llms.abstract.tokenizer import AbstractTokenizerForInference
from tinylm.llms.abstract.generation_config import AbstractGenerationConfig


def model_factory(model: ModelLiteral):
    if model == "unsloth/Llama-3.2-1B-Instruct":
        return Llama3ModelForCasualLM(
            num_layers=16,
            dim=2048,
            ffn_dim=8192,
            kv_heads=8,
            head_dim=64,
            vocab_size=128256,
            rope_theta=500000,
            att_heads=32,
            ctx_len=8192,
        )
    elif model == "unsloth/Qwen3-0.6B":
        return Qwen3ModelForCasualLM(
            num_layers=28,
            dim=1024,
            ffn_dim=3072,
            kv_heads=8,
            head_dim=128,
            vocab_size=151936,
            rope_theta=1000000,
            att_heads=16,
            ctx_len=4096,
        )
    elif model == "llm-jp/llm-jp-3.1-1.8b-instruct4":
        return LlamaModelForCasualLM(
            num_layers=24,
            dim=2048,
            ffn_dim=7168,
            kv_heads=16,
            head_dim=128,
            vocab_size=99584,
            rope_theta=10000,
            att_heads=16,
            ctx_len=4096,
        )


def state_dict_to_dtype(
    state_dict: Dict[str, Tensor], dtype: DType
) -> Dict[str, Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key] = value.to(Device.DEFAULT).cast(dtype)
    return new_state_dict


def load_model(
    console: Console,
    model: ModelLiteral,
    dtype: DType,
) -> Tuple[AbstractTokenizerForInference, LlamaGenerationContext]:
    tokenizer: AbstractTokenizerForInference = AutoTokenizer.from_pretrained(model)
    path = hf_hub_download(model, "model.safetensors")
    try:
        generation_config_path = hf_hub_download(model, "generation_config.json")
        generation_config = AbstractGenerationConfig.model_validate_json(
            open(generation_config_path, "r").read()
        )
    except Exception:
        generation_config = AbstractGenerationConfig()
    generation_config.temperature = 0
    model = model_factory(model)
    state_dict = nn.state.safe_load(path)
    state_dict = state_dict_to_dtype(state_dict, dtype=dtype)
    state_dict["model.rotary_emb.sin"] = model.model.rotary_emb.sin
    state_dict["model.rotary_emb.cos"] = model.model.rotary_emb.cos
    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    for key in state_dict.keys():
        if (
            (
                "self_attn" in key
                and (
                    "q_proj" in key
                    or "k_proj" in key
                    or "v_proj" in key
                    or "o_proj" in key
                )
            )
            or (
                "mlp" in key
                and ("gate_proj" in key or "up_proj" in key or "down_proj" in key)
            )
            or "lm_head" in key
        ):
            state_dict[key] = state_dict[key].T
    if isinstance(model, Qwen3ModelForCasualLM):
        for i in range(len(model.model.layers)):
            q_proj_weight = state_dict.pop(f"model.layers.{i}.self_attn.q_proj.weight")
            k_proj_weight = state_dict.pop(f"model.layers.{i}.self_attn.k_proj.weight")
            v_proj_weight = state_dict.pop(f"model.layers.{i}.self_attn.v_proj.weight")
            qkv_weight = Tensor.cat(q_proj_weight, k_proj_weight, v_proj_weight, dim=-1)
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = qkv_weight
            gate_proj_weight = state_dict.pop(f"model.layers.{i}.mlp.gate_proj.weight")
            up_proj_weight = state_dict.pop(f"model.layers.{i}.mlp.up_proj.weight")
            gate_up_proj_weight = Tensor.cat(gate_proj_weight, up_proj_weight, dim=-1)
            state_dict[f"model.layers.{i}.mlp.gate_up_proj.weight"] = (
                gate_up_proj_weight
            )
    nn.state.load_state_dict(model, state_dict)
    total_params = sum(param.numel() for param in nn.state.get_parameters(model))
    console = Console()
    console.print(
        f"Total parameters in the model: {total_params / 1e6:.2f}M", style="green"
    )
    context = LlamaGenerationContext(
        model,
        batch_size=1,
        prefill_chunk_size=256,
        dtype=dtype,
        pad_token_id=generation_config.pad_token_id or int(tokenizer.pad_token_id),
        eos_token_id=generation_config.eos_token_id or int(tokenizer.eos_token_id),
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
    )
    return tokenizer, context


def chat_main(model: ModelLiteral, dtype: DType):
    console = Console()
    tokenizer, context = load_model(console, model, dtype)
    console.print("Start warmup...", style="green")
    context.warmup()
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
        input_ids = tokenizer(
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
        ).input_ids
        outputs = []
        output_texts = ""
        output_tokens = 0
        prefill_tokens = 0
        prefill_time = -1
        generate_time = -1
        for chunk in context.generate(
            input_ids,
            max_new_tokens=context.model.ctx_len // 2,
        ):
            if chunk.type == "token":
                output_tokens += 1
                outputs.append(chunk.token)  # ty: ignore[possibly-unbound-attribute]
                text = tokenizer.decode(outputs)
                if text.endswith("\n") or text.endswith(" "):
                    output_texts += text
                    outputs = []
                    live.update(styled_markdown(output_texts), refresh=True)
            elif chunk.type == "end":
                output_texts += tokenizer.decode(outputs)
                live.update(styled_markdown(output_texts), refresh=True)
                live.stop()
                generate_time = time.time() - generate_time
                console.print(
                    f"Input tokens: {len(input_ids)}\nOutput tokens: {output_tokens}\nPrefill TPS: {prefill_tokens / prefill_time:.2f}\nGenerate TPS: {output_tokens / generate_time:.2f}",
                    style="green",
                )
                chat.append(
                    {
                        "role": "assistant",
                        "content": output_texts.split("</think>")[-1].strip(),
                    }
                )
            elif chunk.type == "prefill_start":
                prefill_time = time.time()
                prefill_tokens = (
                    chunk.prefill_tokens  # ty: ignore[possibly-unbound-attribute]
                )
            elif chunk.type == "prefill_end":
                prefill_time = time.time() - prefill_time
                generate_time = time.time()
