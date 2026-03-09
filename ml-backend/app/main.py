"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.config import settings
from app.services.instances import worker
from app.services.job_store import job_store

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = set(settings.cors_origins)


class CORSMiddleware:
    """Pure ASGI middleware for CORS — handles preflight and regular requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode()
        method = scope.get("method", "")

        # Handle preflight OPTIONS requests
        if method == "OPTIONS" and origin in ALLOWED_ORIGINS:
            request_headers = headers.get(
                b"access-control-request-headers", b""
            ).decode()
            logger.info("CORS preflight from origin=%s headers=%s", origin, request_headers)
            response_headers = [
                (b"access-control-allow-origin", origin.encode()),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS, PATCH"),
                (b"access-control-allow-headers", request_headers.encode() if request_headers else b"*"),
                (b"access-control-allow-credentials", b"true"),
                (b"access-control-max-age", b"86400"),
                (b"content-length", b"0"),
            ]
            await send({"type": "http.response.start", "status": 200, "headers": response_headers})
            await send({"type": "http.response.body", "body": b""})
            return

        # For non-preflight requests, add CORS headers to the response
        if origin in ALLOWED_ORIGINS:
            cors_headers = [
                (b"access-control-allow-origin", origin.encode()),
                (b"access-control-allow-credentials", b"true"),
            ]

            async def send_with_cors(message):
                if message["type"] == "http.response.start":
                    message["headers"] = list(message.get("headers", [])) + cors_headers
                await send(message)

            await self.app(scope, receive, send_with_cors)
        else:
            await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: recover any jobs stuck from a previous crash, start worker
    recovered = job_store.recover_stale_jobs()
    if recovered:
        logger.info("Recovered %d stale jobs from previous run", recovered)
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
