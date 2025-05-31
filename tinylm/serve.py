from tinylm.clidefs import ModelLiteral
from tinygrad.tensor import DType
from tinylm.chat import load_model
from rich.console import Console
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from uvicorn import run
from openai.resources.models import SyncPage, Model
from openai.resources.chat.completions.completions import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionChunk,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice, ChoiceDelta
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import time
from pydantic import BaseModel
from typing import Iterable, List
from asyncio import Lock
import random


def generate_cmpl_id():
    return "chatcmpl-" + "".join(
        random.choices(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            k=16,
        )
    )


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Iterable[ChatCompletionMessageParam]
    stream: bool = False


def serve_main(model: ModelLiteral, dtype: DType, host: str, port: int):
    console = Console()
    tokenizer, context = load_model(console, model, dtype)
    console.print("Start warmup...", style="green")
    context.warmup()
    console.print("Warmup done.", style="green")
    app = FastAPI()
    lock = Lock()

    @app.get("/models")
    async def ep_models():
        return JSONResponse(
            content=SyncPage(
                data=[
                    Model(
                        id=model,
                        object="model",
                        created=int(time.time()),
                        owned_by="tinylm_serve",
                    ),
                    Model(
                        id="tinylm-auto",
                        object="model",
                        created=int(time.time()),
                        owned_by="tinylm_serve",
                    ),
                ],
                object="list",
            ).to_dict(),
        )

    async def generate_stream(
        input_ids: List[int],
        model: str,
    ):
        index = 0
        generated_tokens = []
        sent_chunks = ""
        for chunk in context.generate(
            input_ids,
            max_new_tokens=context.model.ctx_len // 2,
        ):
            data = None
            if chunk.type == "token":
                generated_tokens.append(
                    chunk.token  # ty: ignore[possibly-unbound-attribute]
                )
                decoded = tokenizer.decode(generated_tokens)
                not_sent_chunks = decoded[len(sent_chunks) :]
                should_send_tokens = None
                if "\n" in not_sent_chunks:
                    should_send_tokens = "\n".join(not_sent_chunks.split("\n")[:-1])
                if " " in not_sent_chunks:
                    should_send_tokens = " ".join(not_sent_chunks.split(" ")[:-1])
                if "　" in not_sent_chunks:
                    should_send_tokens = "　".join(not_sent_chunks.split("　")[:-1])
                if should_send_tokens is not None:
                    sent_chunks += should_send_tokens
                    data = ChatCompletionChunk(
                        created=int(time.time()),
                        choices=[
                            StreamChoice(
                                delta=ChoiceDelta(
                                    role="assistant",
                                    content=should_send_tokens,
                                ),
                                index=index,
                            )
                        ],
                        model=model,
                        id=generate_cmpl_id(),
                        object="chat.completion.chunk",
                    )
                    index += 1
            elif chunk.type == "end":
                data = ChatCompletionChunk(
                    created=int(time.time()),
                    choices=[
                        StreamChoice(
                            delta=ChoiceDelta(
                                role="assistant",
                                content=tokenizer.decode(
                                    generated_tokens,
                                )[len(sent_chunks) :],
                            ),
                            finish_reason="stop"
                            if chunk.reason  # ty: ignore[possibly-unbound-attribute]
                            == "eos"
                            else "length",
                            index=index,
                        ),
                    ],
                    model=model,
                    id=generate_cmpl_id(),
                    object="chat.completion.chunk",
                    usage=CompletionUsage(
                        prompt_tokens=len(input_ids),
                        completion_tokens=len(generated_tokens),
                        total_tokens=len(input_ids) + len(generated_tokens),
                    ),
                )
            if data is not None:
                yield f"data: {data.model_dump_json()}\n\n"

    async def ep_chat_completions_inner(req: ChatCompletionRequest):
        input_ids = tokenizer(
            tokenizer.apply_chat_template(
                [dict(msg) for msg in req.messages],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
        ).input_ids
        if req.stream:
            return StreamingResponse(
                content=generate_stream(input_ids, req.model),
                media_type="text/event-stream",
            )
        else:
            output_ids = []
            reason = "max_new_tokens"
            for chunk in context.generate(
                input_ids,
                max_new_tokens=context.model.ctx_len // 2,
            ):
                if chunk.type == "token":
                    output_ids.append(
                        chunk.token  # ty: ignore[possibly-unbound-attribute]
                    )
                elif chunk.type == "end":
                    reason = chunk.reason  # ty: ignore[possibly-unbound-attribute]
            return JSONResponse(
                content=ChatCompletion(
                    id=generate_cmpl_id(),
                    object="chat.completion",
                    created=int(time.time()),
                    model=req.model,
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=tokenizer.decode(output_ids),
                            ),
                            finish_reason="stop" if reason == "eos" else "length",
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=len(input_ids),
                        completion_tokens=len(output_ids),
                        total_tokens=len(input_ids) + len(output_ids),
                    ),
                ).to_dict(),
            )

    @app.post("/chat/completions")
    async def ep_chat_completions(req: ChatCompletionRequest):
        async with lock:
            return await ep_chat_completions_inner(req)

    run(app, host=host, port=port)
