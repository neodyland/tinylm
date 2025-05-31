FROM ghcr.io/astral-sh/uv:debian-slim
WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
COPY .python-version ./
RUN uv python install
COPY pyproject.toml ./
COPY uv.lock ./
RUN uv sync --no-dev --all-extras --frozen --no-cache
COPY . .
ENTRYPOINT ["uv", "run", "--no-dev", "python", "-m", "tinylm"]