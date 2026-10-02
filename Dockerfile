# Multi-stage build for AegisLLM Gateway
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

# Copy dependency specifications
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies into virtual environment
ENV UV_COMPILE_BYTECODE=1
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache -e .

# Stage 2: Final Minimal Runtime
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy virtualenv and application code
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Setup cache directories with proper permissions
RUN mkdir -p /app/.fastembed_cache && \
    chown -R appuser:appgroup /app

USER appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=7860 \
    FASTEMBED_CACHE_PATH=/app/.fastembed_cache

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "presentation.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2", "--loop", "uvloop"]
