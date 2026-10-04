"""WebSocket endpoint for live frame analysis."""

from __future__ import annotations

import json
import logging
import threading
import time
from io import BytesIO

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.ml.landmarks.session import SessionStats
from app.ml.pipeline import AnalysisPipeline, FrameResult, create_pipeline

router = APIRouter()

logger = logging.getLogger(__name__)

# Close code for "server busy, try again later" (RFC 6455 section 7.4.1).
WS_TRY_AGAIN_LATER = 1013

_sockets_lock = threading.Lock()
_live_sockets = 0


def _admit() -> bool:
    global _live_sockets
    with _sockets_lock:
        if _live_sockets >= settings.max_live_sockets:
            return False
        _live_sockets += 1
        return True


def _release() -> None:
    global _live_sockets
    with _sockets_lock:
        _live_sockets = max(0, _live_sockets - 1)


def live_socket_count() -> int:
    return _live_sockets


def _analyse(
    pipeline: AnalysisPipeline,
    data: bytes,
    prev_frame: np.ndarray | None,
    timestamp: float,
    frame_index: int,
) -> tuple[np.ndarray, FrameResult]:
    """Decode + analyse one JPEG. Runs in the threadpool, off the event loop.

    The PEACE input is built exactly as before (PIL RGB, resize to 224x224);
    the untouched full-resolution frame is only handed on when a landmark
    classifier is attached.
    """
    img = Image.open(BytesIO(data)).convert("RGB")
    frame_array = np.array(img.resize((224, 224)))
    full_frame = np.asarray(img) if pipeline.landmark_classifier is not None else None
    result = pipeline.analyze_frame(
        frame=frame_array,
        prev_frame=prev_frame,
        timestamp=timestamp,
        total_duration=0.0,
        frame_index=frame_index,
        full_frame=full_frame,
    )
    return frame_array, result


async def _send_result(websocket: WebSocket, message: dict) -> None:
    """Send a frame result. A serialisation fault never closes the socket and a
    bad landmark block never costs the PEACE fields."""
    if "landmark" in message:
        try:
            json.dumps(message["landmark"], allow_nan=False)
        except (TypeError, ValueError):
            logger.exception("Dropping unserialisable landmark block")
            message = {k: v for k, v in message.items() if k != "landmark"}
    try:
        await websocket.send_json(message)
        return
    except (TypeError, ValueError):
        logger.exception("Frame result %s failed to serialise", message.get("frame_index"))
    if "landmark" in message:
        try:
            await websocket.send_json({k: v for k, v in message.items() if k != "landmark"})
            return
        except (TypeError, ValueError):
            logger.exception("Frame result failed to serialise without landmark too")
    await websocket.send_json(
        {"type": "error", "frame_index": message.get("frame_index"), "message": "Frame could not be analysed"}
    )


@router.websocket("/api/v1/ws/live")
async def live_analysis(websocket: WebSocket):
    """Stream live frame analysis via WebSocket.

    Client sends binary JPEG frames.
    Server responds with one JSON analysis result per frame.
    """
    if not _admit():
        # Accept, then close with a reason the client can show; a handshake
        # rejection would surface as an opaque connection error.
        await websocket.accept()
        logger.warning("Refusing live socket: %d already open", settings.max_live_sockets)
        await websocket.close(code=WS_TRY_AGAIN_LATER, reason="Too many live sessions; try again shortly")
        return

    stats = SessionStats()
    try:
        await websocket.accept()

        # May load models on first use: keep it off the event loop.
        pipeline = await run_in_threadpool(create_pipeline, with_landmarks=True)
        prev_frame: np.ndarray | None = None
        frame_count = 0

        while True:
            # Receive frame data (binary)
            data = await websocket.receive_bytes()
            start_time = time.time()

            try:
                # Decode + inference are CPU-bound and blocking; off the event
                # loop they go, so one client cannot stall every other connection.
                frame_array, result = await run_in_threadpool(
                    _analyse, pipeline, data, prev_frame, time.time(), frame_count
                )
            except Exception:
                # One unreadable frame must not end the procedure. Report it and
                # keep the socket open; the client can tell a hiccup from a death.
                logger.exception("Frame %s failed to analyse", frame_count)
                stats.record_error()
                await websocket.send_json(
                    {
                        "type": "error",
                        "frame_index": frame_count,
                        "message": "Frame could not be analysed",
                    }
                )
                frame_count += 1
                continue

            processing_time = (time.time() - start_time) * 1000

            message = {
                "type": "frame_result",
                "timestamp": result.timestamp,
                "frame_index": frame_count,
                "peace_score": result.peace_score,
                "motion": result.motion,
                "region": result.region,
                "processing_time_ms": round(processing_time, 1),
            }
            if result.landmark is not None:
                message["landmark"] = result.landmark
            await _send_result(websocket, message)
            stats.record_frame(processing_time, result.landmark)

            prev_frame = frame_array
            frame_count += 1

    except WebSocketDisconnect:
        pass
    finally:
        _release()
        _log_summary(stats)


def _log_summary(stats: SessionStats) -> None:
    """One structured line per session: status histogram, latency, errors."""
    if not stats.frames:
        return
    try:
        logger.info("Live session summary %s", json.dumps(stats.summary(), sort_keys=True, default=str))
    except Exception:
        logger.exception("Could not summarise live session")
