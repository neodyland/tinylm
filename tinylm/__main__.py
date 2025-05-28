from typer import Typer
from tinylm.clidefs import CliDType, Model

app = Typer()


@app.command()
def chat(model: Model, dtype: CliDType = CliDType.float32):
    from .chat import chat_main

    chat_main(model.value, dtype.dtype)


@app.command()
def serve(
    model: Model,
    dtype: CliDType = CliDType.float32,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    from .serve import serve_main

    serve_main(model.value, dtype.dtype, host, port)


@app.command()
def remote_chat(
    model: str = "gpt-1000",
    api_key: str = "hello",
    api_base: str = "http://localhost:8000",
):
    from .remote_chat import remote_chat_main

    remote_chat_main(model, api_key, api_base)


if __name__ == "__main__":
    app()
