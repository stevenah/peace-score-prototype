"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.config import settings
from app.services.instances import worker
from app.services.job_store import job_store

logger = logging.getLogger(__name__)


class PreflightMiddleware(BaseHTTPMiddleware):
    """Handle CORS preflight before any route validation."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            origin = request.headers.get("origin", "")
            if origin in settings.cors_origins:
                return Response(
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                        "Access-Control-Allow-Headers": request.headers.get(
                            "access-control-request-headers", "*"
                        ),
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Max-Age": "86400",
                    },
                )
        return await call_next(request)


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

# PreflightMiddleware runs first (added last = outermost) to catch OPTIONS
# before CORSMiddleware or route handlers can reject them
app.add_middleware(PreflightMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)
