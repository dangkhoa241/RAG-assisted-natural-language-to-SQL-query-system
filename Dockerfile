# One image serves the React build and the FastAPI API on one port (a Hugging Face Docker Space).
#
#   docker build -t nl2sql .
#   docker run -p 7860:7860 -e GROQ_API_KEY=... -e INTENT_MODEL_PATH=user/intent-model nl2sql
#
# Stage 1 builds the frontend with Node; stage 2 is a slim Python image with the runtime requirements only
# (CPU-only torch, no sentence-transformers/FAISS). API keys are never baked in: they come from the
# environment at run time (Space secrets).

# --- 1. Frontend build -------------------------------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


# --- 2. Python runtime -------------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch first, from PyTorch's index, so pip never pulls the multi-GB CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Hugging Face Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860 \
    TRUSTED_PROXY_HOPS=1
WORKDIR /home/user/app

# Optional: download the intent model at build time, so a cold start doesn't wait for it. On Spaces, the
# INTENT_MODEL_PATH variable is passed as a build argument; without it the app loads the model at startup
# (or uses keyword intents if INTENT_MODEL_PATH isn't set at all).
ARG INTENT_MODEL_PATH=""
RUN if [ -n "$INTENT_MODEL_PATH" ]; then \
      python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['INTENT_MODEL_PATH'])" \
      || echo "intent model prefetch skipped; it will load at startup"; \
    fi

COPY --chown=user backend/ backend/
COPY --chown=user src/ src/
COPY --chown=user config/ config/
COPY --chown=user docs/glossary/ docs/glossary/
COPY --chown=user data/healthcare_dataset.csv data/retail_sales.csv data/saas_subscriptions.csv data/
COPY --chown=user --from=frontend /build/dist frontend/dist

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# One worker: uploaded datasets, rate limits and the daily budget live in this process's memory.
# The client IP comes from X-Forwarded-For via TRUSTED_PROXY_HOPS (backend/main.py), not uvicorn's proxy
# headers, so a client can't spoof it by sending its own header.
CMD ["sh", "-c", "exec uvicorn backend.main:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1 --no-proxy-headers --no-server-header"]
