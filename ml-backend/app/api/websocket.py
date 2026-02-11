"""WebSocket endpoint for live frame analysis."""

from __future__ import annotations

import time
from io import BytesIO

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

from app.ml.pipeline import create_pipeline

router = APIRouter()


@router.websocket("/api/v1/ws/live")
async def live_analysis(websocket: WebSocket):
    """Stream live frame analysis via WebSocket.

    Client sends binary JPEG frames.
    Server responds with JSON analysis results.
    """
    await websocket.accept()

    pipeline = create_pipeline()
    prev_frame: np.ndarray | None = None
    frame_count = 0

    try:
        while True:
            # Receive frame data (binary)
            data = await websocket.receive_bytes()
            start_time = time.time()

            # Decode frame
            img = Image.open(BytesIO(data)).convert("RGB")
            frame_array = np.array(img.resize((224, 224)))

            # Analyze
            result = pipeline.analyze_frame(
                frame=frame_array,
                prev_frame=prev_frame,
                timestamp=time.time(),
                total_duration=0.0,
                frame_index=frame_count,
            )

            processing_time = (time.time() - start_time) * 1000

            # Send result
            await websocket.send_json(
                {
                    "type": "frame_result",
                    "timestamp": result.timestamp,
                    "frame_index": frame_count,
                    "peace_score": result.peace_score,
                    "motion": result.motion,
                    "region": result.region,
                    "processing_time_ms": round(processing_time, 1),
                }
            )

            prev_frame = frame_array
            frame_count += 1

    except WebSocketDisconnect:
        pass
