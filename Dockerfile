FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependencies first for Docker layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-cache

# Copy application source code
COPY config/ ./config/
COPY src/ ./src/
COPY server/ ./server/
COPY menu.yaml agent.py main.py ./

# Use shell form of CMD to expand the $PORT environment variable
CMD uv run uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}