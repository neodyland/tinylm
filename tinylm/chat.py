from tinygrad import Tensor, nn, Device
from tinygrad.tensor import DType
from transformers import AutoTokenizer
from typing import Any, Dict, Tuple
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
            ctx_len=40960,
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
    model = model_factory(model)
    state_dict = nn.state.safe_load(path)
    state_dict = state_dict_to_dtype(state_dict, dtype=dtype)
    state_dict["model.rotary_emb.sin"] = model.model.rotary_emb.sin
    state_dict["model.rotary_emb.cos"] = model.model.rotary_emb.cos
    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        top_p=0.8,
        top_k=20,
    )
    return tokenizer, context


@Tensor.test()
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
        prefill_tokens = 0
        prefill_time = -1
        generate_time = -1
        for chunk in context.generate(
            input_ids,
            max_new_tokens=context.model.ctx_len // 2,
        ):
            if chunk.type == "token":
                outputs.append(chunk.token)  # ty: ignore[possibly-unbound-attribute]
                live.update(styled_markdown(tokenizer.decode(outputs)), refresh=True)
            elif chunk.type == "end":
                live.stop()
                generate_time = time.time() - generate_time
                console.print(
                    f"Input tokens: {len(input_ids)}\nOutput tokens: {len(outputs)}\nPrefill TPS: {prefill_tokens / prefill_time:.2f}\nGenerate TPS: {len(outputs) / generate_time:.2f}",
                    style="green",
                )
                chat.append(
                    {
                        "role": "assistant",
                        "content": tokenizer.decode(outputs)
                        .split("</think>")[-1]
                        .strip(),
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
