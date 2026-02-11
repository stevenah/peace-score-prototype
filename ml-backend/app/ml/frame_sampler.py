"""Frame extraction from video files."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ExtractedFrame:
    index: int
    timestamp: float
    data: np.ndarray


def extract_frames(
    video_path: str,
    sample_rate_fps: float = 2.0,
    max_frames: int = 500,
) -> tuple[list[ExtractedFrame], dict]:
    """Extract frames from a video at the given sample rate.

    Returns:
        Tuple of (list of frames, video metadata dict)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    metadata = {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": (width, height),
        "total_frames": total_frames,
    }

    # Calculate frame interval
    frame_interval = max(1, int(fps / sample_rate_fps))

    frames: list[ExtractedFrame] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0 and len(frames) < max_frames:
            timestamp = frame_idx / fps if fps > 0 else 0
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize for ML model input (224x224)
            frame_resized = cv2.resize(frame_rgb, (224, 224))

            frames.append(
                ExtractedFrame(
                    index=frame_idx,
                    timestamp=round(timestamp, 2),
                    data=frame_resized,
                )
            )

        frame_idx += 1

    cap.release()

    metadata["analyzed_frames"] = len(frames)
    return frames, metadata
