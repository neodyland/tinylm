from tinygrad import Tensor, nn
from transformers import AutoTokenizer
from typing import Any
from huggingface_hub import hf_hub_download
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from tinylm.models.llama3.model import Llama3ModelForCasualLM
from tinylm.models.llama.generation_context import LlamaGenerationContext


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
    tokenizer: Any = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    model = Llama3ModelForCasualLM(
        num_layers=16,
        dim=2048,
        ffn_dim=8192,
        kv_heads=8,
        head_dim=64,
        vocab_size=128256,
        rope_theta=500000.0,
        att_heads=32,
        ctx_len=8192,
    )
    path = hf_hub_download("unsloth/Llama-3.2-1B-Instruct", "model.safetensors")
    state_dict = nn.state.safe_load(path)
    state_dict["model.rotary_emb.sin"] = model.model.rotary_emb.sin
    state_dict["model.rotary_emb.cos"] = model.model.rotary_emb.cos
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
        dtype=state_dict["model.embed_tokens.weight"].dtype,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        top_p=0.8,
        top_k=20,
    )
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
            max_new_tokens=model.ctx_len // 2,
        ):
            if chunk.type == "token":
                outputs.append(chunk.token)
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
                prefill_tokens = chunk.prefill_tokens
            elif chunk.type == "prefill_end":
                prefill_time = time.time() - prefill_time
                generate_time = time.time()


if __name__ == "__main__":
    main()
