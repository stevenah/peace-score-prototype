"""HTTP API routes for the ML backend."""

from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from PIL import Image

from app.api.schemas import (
    AnalysisResponse,
    AnalysisStatus,
    FrameAnalysisResponse,
    HealthResponse,
)
from app.config import settings
from app.ml.frame_sampler import extract_frames
from app.ml.pipeline import create_pipeline
from app.services.task_service import task_store

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        gpu_available=False,
        version=settings.version,
        use_mock=settings.use_mock_models,
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
):
    """Upload a video for async analysis."""
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

    # Save uploaded file
    os.makedirs(settings.upload_dir, exist_ok=True)
    task_id = task_store.create_task()
    file_path = os.path.join(settings.upload_dir, f"{task_id}.mp4")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Start background analysis
    background_tasks.add_task(_run_analysis, task_id, file_path)

    task = task_store.get_task(task_id)
    return {
        "analysis_id": task_id,
        "status": "queued",
        "estimated_duration_seconds": 30,
        "created_at": task["created_at"] if task else "",
    }


@router.get("/analyze/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Poll for analysis status and results."""
    task = task_store.get_task(analysis_id)
    if not task:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResponse(**task)


def _run_analysis(task_id: str, file_path: str) -> None:
    """Background task to run the full video analysis pipeline."""
    try:
        task_store.update_task(
            task_id, status=AnalysisStatus.PROCESSING, progress=0.1
        )

        # Extract frames
        frames, metadata = extract_frames(
            file_path, sample_rate_fps=settings.sample_rate_fps
        )
        task_store.update_task(task_id, progress=0.3)

        # Run pipeline
        pipeline = create_pipeline()
        frame_tuples = [(f.index, f.timestamp, f.data) for f in frames]
        results = pipeline.analyze_frames(frame_tuples, metadata["duration_seconds"])
        task_store.update_task(task_id, progress=0.8)

        # Aggregate
        aggregated = pipeline.aggregate_results(results, metadata["duration_seconds"])
        task_store.update_task(task_id, progress=0.95)

        # Complete
        task_store.complete_task(task_id, results=aggregated, metadata=metadata)

    except Exception as e:
        task_store.fail_task(task_id, str(e))
    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass
