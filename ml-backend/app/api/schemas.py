from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PeaceScoreValue(int, Enum):
    POOR = 0
    INADEQUATE = 1
    ADEQUATE = 2
    EXCELLENT = 3


class AnatomicalRegion(str, Enum):
    ESOPHAGUS = "esophagus"
    STOMACH = "stomach"
    DUODENUM = "duodenum"


class MotionDirection(str, Enum):
    INSERTION = "insertion"
    RETRACTION = "retraction"
    STATIONARY = "stationary"


class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Request schemas ---


class AnalysisConfig(BaseModel):
    sample_rate_fps: float = 2.0
    enable_motion_detection: bool = True
    enable_peace_scoring: bool = True
    regions: list[AnatomicalRegion] = [
        AnatomicalRegion.ESOPHAGUS,
        AnatomicalRegion.STOMACH,
        AnatomicalRegion.DUODENUM,
    ]


class FrameContext(BaseModel):
    region_hint: Optional[AnatomicalRegion] = None
    include_motion: bool = True


# --- Response schemas ---


class PeaceScoreResult(BaseModel):
    score: int = Field(ge=0, le=3)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class RegionScore(PeaceScoreResult):
    region: AnatomicalRegion
    frame_scores: list[FrameScoreEntry] = []


class FrameScoreEntry(BaseModel):
    frame_index: int
    timestamp: float
    score: int = Field(ge=0, le=3)
    confidence: float


class MotionSegment(BaseModel):
    start_time: float
    end_time: float
    direction: MotionDirection
    confidence: float


class MotionResult(BaseModel):
    direction: MotionDirection
    confidence: float
    optical_flow_magnitude: float = 0.0


class VideoMetadata(BaseModel):
    duration_seconds: float
    fps: float
    resolution: tuple[int, int]
    total_frames: int
    analyzed_frames: int


class TimelineEntry(BaseModel):
    timestamp: float
    frame_index: int
    motion: MotionDirection
    region: AnatomicalRegion
    peace_score: int = Field(ge=0, le=3)
    confidence: float


class MotionAnalysis(BaseModel):
    segments: list[MotionSegment]


class PeaceScores(BaseModel):
    overall: PeaceScoreResult
    by_region: dict[AnatomicalRegion, RegionScore]


class AnalysisResults(BaseModel):
    motion_analysis: MotionAnalysis
    peace_scores: PeaceScores
    timeline: list[TimelineEntry]


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    progress: float = 0.0
    video_metadata: Optional[VideoMetadata] = None
    results: Optional[AnalysisResults] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class FrameAnalysisResponse(BaseModel):
    peace_score: PeaceScoreResult
    motion: Optional[MotionResult] = None
    region: Optional[AnatomicalRegion] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    gpu_available: bool
    version: str
    use_mock: bool


class LiveFrameResult(BaseModel):
    type: str = "frame_result"
    timestamp: float
    frame_index: int
    peace_score: PeaceScoreResult
    motion: Optional[MotionResult] = None
    region: Optional[AnatomicalRegion] = None
    processing_time_ms: float


# Forward reference resolution
RegionScore.model_rebuild()
