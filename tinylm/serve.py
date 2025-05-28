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
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice, ChoiceDelta
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
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
        model: ModelLiteral,
    ):
        index = 0
        for chunk in context.generate(
            input_ids,
            max_new_tokens=context.model.ctx_len // 2,
        ):
            data = None
            if chunk.type == "token":
                data = ChatCompletionChunk(
                    index=index,
                    created=int(time.time()),
                    choices=[
                        StreamChoice(
                            delta=ChoiceDelta(
                                role="assistant",
                                content=tokenizer.decode([chunk.token]),
                            ),
                            index=index,
                        )
                    ],
                    model=model,
                    id=generate_cmpl_id(),
                    object="chat.completion.chunk",
                )
            elif chunk.type == "end":
                data = ChatCompletionChunk(
                    index=index,
                    created=int(time.time()),
                    choices=[
                        StreamChoice(
                            delta=ChoiceDelta(),
                            finish_reason="stop" if chunk.reason == "eos" else "length",
                            index=index,
                        ),
                    ],
                    model=model,
                    id=generate_cmpl_id(),
                    object="chat.completion.chunk",
                )
            if data is not None:
                yield f"data: {data.model_dump_json()}\n\n"
            index += 1

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
                max_new_tokens=model.ctx_len // 2,
            ):
                if chunk.type == "token":
                    output_ids.append(chunk.token)
                elif chunk.type == "end":
                    reason = chunk.reason
            return JSONResponse(
                content=ChatCompletion(
                    id=generate_cmpl_id(),
                    object="chat.completion",
                    created=int(time.time()),
                    model=req.model,
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=tokenizer.decode(output_ids),
                            ),
                            finish_reason="stop" if reason == "eos" else "length",
                        )
                    ],
                ).to_dict(),
            )

    @app.post("/chat/completions")
    async def ep_chat_completions(req: ChatCompletionRequest):
        async with lock:
            return await ep_chat_completions_inner(req)

    run(app, host=host, port=port)
