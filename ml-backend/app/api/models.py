"""Model admin API: which landmark bundle is served, and evals of bundles and runs.

    GET    /api/v1/models             bundles, training runs, serving state, eval kinds
    POST   /api/v1/models/active      serve another bundle            {"version": "lm-1.0.0"}
    DELETE /api/v1/models/active      back to the configured default
    PUT    /api/v1/models/mode        off | shadow | display          {"mode": "shadow"}
    GET    /api/v1/evals              recent eval jobs
    POST   /api/v1/evals              start one      {"kind", "target", "params"}
    GET    /api/v1/evals/{id}         one job with its result and log tail
    POST   /api/v1/evals/{id}/cancel

Everything needs the shared API key. The routes that change something also
need ``settings.model_admin``; with it off they answer 403 and the GETs still
describe what is being served.

A switch applies to live sessions opened from then on. Sessions already open
keep the model and mode they started with.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.auth import require_api_key
from app.config import settings
from app.ml.landmarks import registry
from app.ml.landmarks.bundle import LandmarkUnavailable, read_lock
from app.ml.landmarks.classifier import LandmarkModelManager
from app.ml.landmarks.stations import STATION_ORDER
from app.services.evals import EvalError, describe_kinds
from app.services.instances import eval_runner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


def require_model_admin() -> None:
    if not settings.model_admin:
        raise HTTPException(status_code=403, detail="Model management is switched off on this backend")


_admin = [Depends(require_model_admin)]


class ActivateRequest(BaseModel):
    version: str


class ModeRequest(BaseModel):
    mode: Literal["off", "shadow", "display"]


class EvalRequest(BaseModel):
    kind: str
    target: str
    params: dict = Field(default_factory=dict)


def _default_bundle() -> dict:
    """What is served when the admin page has chosen nothing."""
    if settings.landmark_bundle_dir:
        info = registry.describe(settings.landmark_bundle_dir)
        return {"kind": "env", "version": info.version}
    try:
        return {"kind": "lock", "version": read_lock(settings.landmark_bundle_lock).version}
    except LandmarkUnavailable:
        return {"kind": "none", "version": None}


def _serving() -> dict:
    state = registry.read_state()
    manager = LandmarkModelManager.peek()
    return {
        "mode": registry.current_mode(),
        "mode_saved": state.mode is not None,
        "mock": settings.use_mock_models,
        "selected_version": state.version,
        "default": _default_bundle(),
        "loaded_version": manager.version if manager is not None else None,
        "device": str(manager.device) if manager is not None else None,
        "failure": LandmarkModelManager.failure(),
    }


def _load_configured() -> None:
    """Load whatever is now configured, so the answer shows a model or the reason there is none."""
    if settings.landmarks_enabled and not settings.use_mock_models:
        try:
            LandmarkModelManager.get_instance()
        except LandmarkUnavailable:
            pass  # reported as serving.failure


@router.get("/models")
async def list_models():
    def build() -> dict:
        return {
            "admin_enabled": settings.model_admin,
            "serving": _serving(),
            "stations": list(STATION_ORDER),
            "models": [b.summary() for b in registry.discover()],
            "experiments": registry.discover_experiments(),
            "eval_kinds": describe_kinds(eval_runner.kinds),
        }

    return await run_in_threadpool(build)


@router.post("/models/active", dependencies=_admin)
async def activate_model(body: ActivateRequest):
    if settings.use_mock_models:
        raise HTTPException(status_code=409, detail="This backend runs mock models (PEACE_USE_MOCK_MODELS)")

    def activate() -> dict:
        info = registry.find(body.version)
        if info is None:
            raise HTTPException(status_code=404, detail=f"No bundle {body.version!r}")
        try:
            LandmarkModelManager.activate(info.path)
        except LandmarkUnavailable as exc:
            # Nothing was swapped or saved: the previous model is still served.
            reason = str(exc).removeprefix(f"landmark bundle {info.path}: ")
            raise HTTPException(status_code=409, detail=f"{info.version} cannot be served: {reason}") from exc
        state = registry.read_state()
        registry.write_state(registry.ServingState(info.version, str(info.path.resolve()), state.mode))
        logger.info("Landmark bundle %s activated from the admin page", info.version)
        return _serving()

    return await run_in_threadpool(activate)


@router.delete("/models/active", dependencies=_admin)
async def clear_active_model():
    def clear() -> dict:
        registry.write_state(registry.ServingState(mode=registry.read_state().mode))
        LandmarkModelManager.reset()
        _load_configured()
        return _serving()

    return await run_in_threadpool(clear)


@router.put("/models/mode", dependencies=_admin)
async def set_mode(body: ModeRequest):
    def apply() -> dict:
        state = registry.read_state()
        registry.write_state(registry.ServingState(state.version, state.path, body.mode))
        registry.apply_mode(body.mode)
        if LandmarkModelManager.peek() is None:
            LandmarkModelManager.reset()  # a failure cached while landmarks were off may be stale
            _load_configured()
        return _serving()

    return await run_in_threadpool(apply)


@router.get("/evals")
async def list_evals(limit: int = 50):
    return {"jobs": await run_in_threadpool(eval_runner.list, max(1, min(limit, 200)))}


@router.post("/evals", status_code=201, dependencies=_admin)
async def start_eval(body: EvalRequest):
    try:
        return await run_in_threadpool(eval_runner.submit, body.kind, body.target, body.params)
    except EvalError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@router.get("/evals/{job_id}")
async def get_eval(job_id: str):
    job = await run_in_threadpool(eval_runner.get, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Eval not found")
    return job


@router.post("/evals/{job_id}/cancel", dependencies=_admin)
async def cancel_eval(job_id: str):
    job = await run_in_threadpool(eval_runner.cancel, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Eval not found")
    return job
