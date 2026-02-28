"""HTTP API routes for the ML backend."""

from __future__ import annotations

import os
import time
import uuid
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

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


@router.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
):
    """Upload a video for async analysis. The worker thread picks up the job."""
    # Validate file type (check both content_type and extension)
    allowed_types = {
        "video/mp4",
        "video/avi",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
        "application/octet-stream",  # Common for programmatic uploads
    }
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    filename = file.filename or ""
    ext = filename[filename.rfind("."):].lower() if "." in filename else ""
    if file.content_type not in allowed_types and ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: mp4, avi, mov, mkv",
        )

    # Save uploaded file, then create the job
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{file_id}.mp4")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    job_id = job_store.create_job(file_path)

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
