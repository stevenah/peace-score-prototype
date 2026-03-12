"""Frame extraction from video files."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    index: int
    timestamp: float
    data: np.ndarray


def _get_video_metadata(video_path: str) -> dict:
    """Probe video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "csv=p=0:s=,",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

        # Parse stream info (first line): width,height,r_frame_rate,nb_frames,duration
        stream_parts = lines[0].split(",") if lines else []
        width = int(stream_parts[0]) if len(stream_parts) > 0 and stream_parts[0].isdigit() else 0
        height = int(stream_parts[1]) if len(stream_parts) > 1 and stream_parts[1].isdigit() else 0

        # Parse frame rate (e.g. "30/1" or "30000/1001")
        fps = 30.0
        if len(stream_parts) > 2 and "/" in stream_parts[2]:
            num, den = stream_parts[2].split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0

        # Duration from stream or format line
        duration = 0.0
        for part in stream_parts[3:]:
            try:
                duration = float(part)
                if duration > 0:
                    break
            except (ValueError, IndexError):
                continue
        # Fallback: format duration on second line
        if duration <= 0 and len(lines) > 1:
            try:
                duration = float(lines[1])
            except ValueError:
                pass

        total_frames = int(fps * duration) if duration > 0 else 0

        return {
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "resolution": (width, height),
            "total_frames": total_frames,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as exc:
        logger.warning("ffprobe failed, falling back to OpenCV: %s", exc)
        return {}


def _extract_with_ffmpeg(
    video_path: str,
    sample_rate_fps: float,
    max_frames: int,
    metadata: dict,
) -> list[ExtractedFrame]:
    """Extract frames using FFmpeg subprocess (handles HEVC reliably)."""
    width, height = metadata.get("resolution", (0, 0))

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "warning",
        "-i", video_path,
        "-vf", f"fps={sample_rate_fps},scale=224:224",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-",
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    frame_size = 224 * 224 * 3
    frames: list[ExtractedFrame] = []
    fps = metadata.get("fps", 30.0)
    source_interval = fps / sample_rate_fps if sample_rate_fps > 0 else 1.0

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            if max_frames > 0 and len(frames) >= max_frames:
                break

            frame_data = np.frombuffer(raw, dtype=np.uint8).reshape(224, 224, 3)
            source_idx = int(len(frames) * source_interval)
            timestamp = source_idx / fps if fps > 0 else 0.0

            frames.append(
                ExtractedFrame(
                    index=source_idx,
                    timestamp=round(timestamp, 2),
                    data=frame_data.copy(),
                )
            )
    finally:
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()

    return frames


def _extract_with_opencv(
    video_path: str,
    sample_rate_fps: float,
    max_frames: int,
) -> tuple[list[ExtractedFrame], dict]:
    """Fallback: extract frames using OpenCV (less reliable for HEVC)."""
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

    frame_interval = max(1, int(fps / sample_rate_fps))
    frames: list[ExtractedFrame] = []
    frame_idx = 0
    consecutive_failures = 0

    while consecutive_failures < 30:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            frame_idx += 1
            if frame_idx >= total_frames and total_frames > 0:
                break
            continue

        consecutive_failures = 0

        if frame_idx % frame_interval == 0 and (max_frames <= 0 or len(frames) < max_frames):
            timestamp = frame_idx / fps if fps > 0 else 0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


def extract_frames(
    video_path: str,
    sample_rate_fps: float = 2.0,
    max_frames: int = 0,
) -> tuple[list[ExtractedFrame], dict]:
    """Extract frames from a video at the given sample rate.

    Uses FFmpeg subprocess for reliable HEVC/H.265 decoding,
    falling back to OpenCV if FFmpeg is unavailable.

    Returns:
        Tuple of (list of frames, video metadata dict)
    """
    # Try FFmpeg pipeline (handles HEVC properly)
    metadata = _get_video_metadata(video_path)
    if metadata and metadata.get("duration_seconds", 0) > 0:
        try:
            frames = _extract_with_ffmpeg(video_path, sample_rate_fps, max_frames, metadata)
            if frames:
                metadata["analyzed_frames"] = len(frames)
                logger.info(
                    "Extracted %d frames via FFmpeg from %s", len(frames), video_path,
                )
                return frames, metadata
        except Exception as exc:
            logger.warning("FFmpeg extraction failed, falling back to OpenCV: %s", exc)

    # Fallback to OpenCV (with skip-on-failure instead of break)
    logger.info("Using OpenCV fallback for %s", video_path)
    return _extract_with_opencv(video_path, sample_rate_fps, max_frames)
