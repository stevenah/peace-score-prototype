from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ml.landmarks.stations import NUM_STATIONS, SCHEMA, STATION_ORDER


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
    WITHDRAWAL = "withdrawal"
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
    score: float = Field(ge=0, le=3)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    # Present when the real models are in use; the mock backend omits them.
    probs: Optional[list[float]] = None
    region_confidence: Optional[float] = None


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
    video_path: Optional[str] = None
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
    worker_alive: bool = True
    landmarks_loaded: bool = False
    landmark_model_version: Optional[str] = None


# --- ESGE landmark stations (wire schema esge10.v1) ---
# contracts/live_frame_result.v2.example.json is the reference instance.

# Literal over the tuple == Literal["esophagus_proximal", ...] in ESGE order.
StationKey = Literal[STATION_ORDER]  # type: ignore[valid-type]
StationState = Literal["unseen", "candidate", "observed"]
LandmarkStatus = Literal["ok", "uncertain", "low_quality", "unsupported_layout", "skipped", "error"]
ImagingMode = Literal["wl", "nbi", "rdi"]

_CLASSIFIED = ("ok", "uncertain", "low_quality")
_CLASSIFIED_FIELDS = ("probs", "top", "confidence", "quality")


class LandmarkEvents(BaseModel):
    observed: list[StationKey] = []
    best_frame: list[StationKey] = []


class LandmarkResult(BaseModel):
    """The ``landmark`` block of a live frame result (websocket shape).

    Absent from the message when the feature is off. Field presence:
    probs/top/confidence/quality iff status is ok|uncertain|low_quality;
    layout/mode iff the layout was recognised; the rest always.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_: Literal[SCHEMA] = Field(alias="schema")  # type: ignore[valid-type]
    model_version: str
    display: bool
    status: LandmarkStatus
    layout: Optional[str] = None
    mode: Optional[ImagingMode] = None
    probs: Optional[list[float]] = Field(None, min_length=NUM_STATIONS, max_length=NUM_STATIONS)
    top: Optional[StationKey] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    current: Optional[StationKey]
    stations: list[StationState] = Field(min_length=NUM_STATIONS, max_length=NUM_STATIONS)
    auto_enabled: list[bool] = Field(min_length=NUM_STATIONS, max_length=NUM_STATIONS)
    events: LandmarkEvents
    landmark_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _presence_rules(self) -> "LandmarkResult":
        present = self.model_fields_set
        classified = self.status in _CLASSIFIED
        for name in _CLASSIFIED_FIELDS:
            ok = getattr(self, name) is not None if classified else name not in present
            if not ok:
                raise ValueError(f"{name} must be present iff status is one of {_CLASSIFIED}")
        if ("layout" in present) != ("mode" in present) or (
            "layout" in present and (self.layout is None or self.mode is None)
        ):
            raise ValueError("layout and mode are present (non-null) together or not at all")
        if self.probs is not None and not all(0.0 <= p <= 1.0 for p in self.probs):
            raise ValueError("probs must be in [0, 1]")
        return self


class LiveFrameResult(BaseModel):
    type: str = "frame_result"
    timestamp: float
    frame_index: int
    peace_score: PeaceScoreResult
    motion: Optional[MotionResult] = None
    region: Optional[AnatomicalRegion] = None
    processing_time_ms: float
    landmark: Optional[LandmarkResult] = None


# Forward reference resolution
RegionScore.model_rebuild()
