# ──────────────────────────────────────────────────────────────────────────────
# Stock Analyst — production Docker image
#
# Build:  docker compose build
# Run:    docker compose up -d
# ──────────────────────────────────────────────────────────────────────────────

# Use the official slim Python image (matches project's requires-python >=3.11)
FROM python:3.13-slim

# Install uv — fast Python package manager (matches local dev toolchain)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System dependencies needed by some Python packages at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install dependencies (layer cached until pyproject.toml / uv.lock change) ─
COPY pyproject.toml uv.lock ./
# --no-dev: skip pytest / ruff in production image
# --frozen: fail if uv.lock is out of sync (reproducible builds)
RUN uv sync --frozen --no-dev

# ── Copy application source ──────────────────────────────────────────────────
COPY src/       src/
COPY scripts/   scripts/

# ── Create data directory (scan state / scheduler state are stored here) ─────
RUN mkdir -p data

# ── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["uv", "run", "python", "scripts/scheduler.py"]
