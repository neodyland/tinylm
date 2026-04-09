from typing import List
import time
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionMessageParam,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live


def styled_markdown(text: str) -> Markdown:
    text = text.replace("<think>\n\n</think>", "")
    text = text.replace("\n", "\n\n")
    if "</think>" in text:
        parts = text.split("</think>")
        thinking = parts[0].split("<think>")[-1].strip()
        answer = parts[1].strip()
        thinking = f"> {thinking.replace('\n\n', '\n>\n> ')}"
        text = f"{thinking}\n\n{answer}"
    return Markdown(text)


def remove_reasoning(text: str) -> str:
    while True:
        start = text.find("<think>")
        end = text.find("</think>")
        if start == -1 or end == -1:
            break
        text = text[:start] + text[end + len("</think>") :]
    return text


def remote_chat_main(model: str, api_key: str, api_base: str):
    ai = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )

    console = Console()
    console.print(f"Using model: {model}", style="green")
    chat: List[ChatCompletionMessageParam] = []

    while True:
        console.print("User: ", style="cyan", end="")
        user_input = console.input().strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            chat.clear()
            console.print("Chat history cleared.", style="green")
            continue

        chat.append(ChatCompletionUserMessageParam(content=user_input, role="user"))
        console.print("Assistant: ", style="cyan")

        live = Live(console=console, refresh_per_second=4)
        live.start()
        start_time = time.time()
        text = b""
        is_reasoning = False

        response = ai.chat.completions.create(model=model, messages=chat, stream=True)

        for chunk in response:
            delta = chunk.choices[0].delta
            token = delta.content
            reasoning_token = getattr(delta, "reasoning_content", None)
            if reasoning_token is not None:
                if is_reasoning:
                    token = reasoning_token
                else:
                    token = f"\n<think>\n{reasoning_token}"
                    is_reasoning = True
            elif reasoning_token is None and token and is_reasoning:
                token = f"\n</think>\n{token}"
                is_reasoning = False
            if token:
                text += token.encode("utf-8")
                live.update(styled_markdown(text.decode("utf-8")))

        live.stop()
        elapsed = time.time() - start_time
        console.print(f"\nResponse received in {elapsed:.2f}s", style="green")

        chat.append(
            ChatCompletionAssistantMessageParam(
                content=remove_reasoning(text.decode("utf-8")), role="assistant"
            )
        )
