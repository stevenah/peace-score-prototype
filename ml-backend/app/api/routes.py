"""HTTP API routes for the ML backend."""

from __future__ import annotations

import shutil
import time
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.api.schemas import (
    AnalysisResponse,
    FrameAnalysisResponse,
    HealthResponse,
)
from app.config import settings
from app.ml.pipeline import create_pipeline
from app.services.instances import worker
from app.services.job_store import job_store

router = APIRouter(prefix="/api/v1")


def _check_disk_space() -> None:
    """Raise 507 if free disk space is below the configured threshold."""
    try:
        usage = shutil.disk_usage(settings.upload_dir)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < settings.min_free_disk_mb:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient storage ({int(free_mb)}MB free, need {settings.min_free_disk_mb}MB)",
            )
    except OSError:
        pass  # Can't stat — let the write attempt fail naturally


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if worker.is_alive else "degraded",
        models_loaded=True,
        gpu_available=False,
        version=settings.version,
        use_mock=settings.use_mock_models,
        worker_alive=worker.is_alive,
    )


@router.post("/analyze/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(
    frame: UploadFile = File(...),
    previous_frame: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
):
    """Analyze a single frame for PEACE score and motion."""
    start_time = time.time()

    # Read and decode the frame
    frame_bytes = await frame.read()
    img = Image.open(BytesIO(frame_bytes)).convert("RGB")
    frame_array = np.array(img.resize((224, 224)))

    # Read previous frame if provided
    prev_array = None
    if previous_frame:
        prev_bytes = await previous_frame.read()
        prev_img = Image.open(BytesIO(prev_bytes)).convert("RGB")
        prev_array = np.array(prev_img.resize((224, 224)))

    pipeline = create_pipeline()
    result = pipeline.analyze_frame(
        frame=frame_array,
        prev_frame=prev_array,
        timestamp=0.0,
        total_duration=0.0,
        frame_index=0,
    )

    processing_time = (time.time() - start_time) * 1000

    return FrameAnalysisResponse(
        peace_score=result.peace_score,
        motion=result.motion,
        region=result.region,
        processing_time_ms=round(processing_time, 1),
    )


class AnalyzeS3Request(BaseModel):
    s3_key: str


@router.post("/analyze/s3")
async def analyze_from_s3(body: AnalyzeS3Request):
    """Create an analysis job for a video already uploaded to S3.

    The worker thread will download the file from S3 before processing.
    """
    job_id = job_store.create_s3_job(body.s3_key)
    job = job_store.get_job(job_id)
    return {
        "analysis_id": job_id,
        "status": "queued",
        "estimated_duration_seconds": 30,
        "created_at": job["created_at"] if job else "",
    }


@router.get("/analyze/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Poll for analysis status and results."""
    job = job_store.get_job(analysis_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResponse(**job)
