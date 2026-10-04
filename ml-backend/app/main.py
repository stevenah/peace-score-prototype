"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.config import settings
from app.ml.pipeline import warm_up_models
from app.services.instances import worker
from app.services.job_store import job_store

logger = logging.getLogger(__name__)

# uvicorn configures only its own loggers; without a root handler the app's
# INFO lines (effective settings, warm-up, live session summaries) are lost.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

class CORSMiddleware:
    """Pure ASGI middleware — allow all origins."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"*").decode()
        method = scope.get("method", "")

        if method == "OPTIONS":
            request_headers = headers.get(
                b"access-control-request-headers", b""
            ).decode()
            logger.info("CORS preflight from origin=%s headers=%s", origin, request_headers)
            response_headers = [
                (b"access-control-allow-origin", b"*"),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS, PATCH"),
                (b"access-control-allow-headers", request_headers.encode() if request_headers else b"*"),
                (b"access-control-max-age", b"86400"),
                (b"content-length", b"0"),
            ]
            await send({"type": "http.response.start", "status": 200, "headers": response_headers})
            await send({"type": "http.response.body", "body": b""})
            return

        cors_headers = [
            (b"access-control-allow-origin", b"*"),
        ]

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + cors_headers
            await send(message)

        await self.app(scope, receive, send_with_cors)


def _configure_torch() -> None:
    if settings.torch_threads > 0:
        import torch

        torch.set_num_threads(settings.torch_threads)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Effective settings: %s", settings.public_summary())
    _configure_torch()
    # Startup: recover any jobs stuck from a previous crash, start worker
    recovered = job_store.recover_stale_jobs()
    if recovered:
        logger.info("Recovered %d stale jobs from previous run", recovered)
    # Load the models before the worker (or any socket) can race to do it.
    # A landmark failure is logged and leaves /health landmarks_loaded=false.
    if settings.warmup_models:
        logger.info("Model warm-up: %s", warm_up_models())
    worker.start()
    yield
    # Shutdown: graceful stop
    worker.stop()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware)

app.include_router(api_router)
app.include_router(ws_router)
