FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# CPU-only torch, pinned to the versions in uv.lock. The default PyPI linux
# wheel is the CUDA build (GBs of unused libraries, slow cold starts) and an
# unpinned install lets serving drift from what the models were trained with.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.10.*" "torchvision==0.25.*" \
    && pip install --no-cache-dir ".[ml]"

COPY app/ app/

EXPOSE 8000

# --- Dev stage: uvicorn with auto-reload ---
FROM base AS dev
RUN pip install --no-cache-dir watchfiles
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# --- Prod stage ---
FROM base AS prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
