# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage build: a single image that all four services (api, dashboard,
# ingester, scorer) run from — each with a different command. The image is
# distroless-ish (slim base + non-root user + minimal apt) and bakes the
# baseline model artifact at build time for reproducibility.
# ---------------------------------------------------------------------------

# ---- Stage 1: build dependencies into a venv ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

# IMPORTANT: WORKDIR must match the runtime WORKDIR. ``uv sync`` writes an
# editable install pointing at this directory's ``src/``; if the runtime
# stage uses a different path, the .pth file dangles.
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- Stage 2: runtime with model artifact baked in ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Minimal runtime deps — psycopg needs libpq.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 anomaly

WORKDIR /app

# Application code + venv. Path matches the builder so editable install
# references in .venv stay valid.
COPY --from=builder /app/.venv /app/.venv
COPY --chown=anomaly:anomaly src ./src
COPY --chown=anomaly:anomaly dashboard ./dashboard
COPY --chown=anomaly:anomaly scripts ./scripts
COPY --chown=anomaly:anomaly alembic ./alembic
COPY --chown=anomaly:anomaly alembic.ini pyproject.toml ./
COPY --chown=anomaly:anomaly docker/entrypoint-api.sh ./entrypoint-api.sh
RUN chmod +x ./entrypoint-api.sh

# Bake a baseline model into the image so the API can serve immediately on
# first boot. Re-trained on every build, so the model_version reflects the
# build (not whichever artifact someone happened to mount).
RUN mkdir -p /app/models \
    && python scripts/train_baseline.py \
        --out /app/models/baseline.joblib \
        --version "baked-$(date -u +%Y%m%d-%H%M%S)" \
    && chown -R anomaly:anomaly /app/models

USER anomaly

# Sensible defaults; compose / k8s override what they need.
ENV API_MODEL_PATH=/app/models/baseline.joblib \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

EXPOSE 8000 8501

# Default to the API role — migrations then uvicorn. docker-compose overrides
# both ENTRYPOINT and CMD per service for dashboard / ingester / scorer; cloud
# runtimes (App Runner / Container Apps) need a real default here or they
# launch the container with nothing to run and exit immediately.
ENTRYPOINT ["./entrypoint-api.sh"]
CMD ["uvicorn", "factory_anomaly.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
