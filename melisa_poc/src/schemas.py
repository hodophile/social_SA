"""Pydantic schemas for MyUni sentiment analysis POC inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


SentimentLabel = Literal["positive", "neutral", "negative"]
ActivityType = Literal["text", "image", "video", "audio"]

# Reserved for future MyUni semantics (post / comment / story). Not enforced yet.
ContentKind = Literal["post", "comment", "story", "caption", "other"]

BatchRecordStatus = Literal["processed", "invalid", "unsupported", "failed", "skipped"]


# ---------------------------------------------------------------------------
# Activity input contract (batch / daily workflow)
# ---------------------------------------------------------------------------


class ActivityInput(BaseModel):
    """Validated MyUni activity record for ingestion.

    Batch validation policy (Milestone 2):
    - ``activity_id`` and ``user_id`` are required non-blank strings
    - ``text`` activities require usable (non-blank) ``text``
    - ``image`` / ``video`` activities require ``media_path``; caption ``text`` is optional
    - ``content_kind`` is optional reserved extensibility (post/comment/story), unused by MVP logic
    """

    activity_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    activity_type: ActivityType
    text: Optional[str] = None
    media_path: Optional[str] = None
    created_at: datetime
    metadata: Optional[dict[str, Any]] = None
    content_kind: Optional[ContentKind] = Field(
        default=None,
        description="Optional future semantic kind (post/comment/story); ignored by MVP routing.",
    )

    @field_validator("activity_id", "user_id", mode="before")
    @classmethod
    def _strip_required_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("media_path", mode="before")
    @classmethod
    def _strip_optional_path(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("text", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def _enforce_modality_rules(self) -> ActivityInput:
        if self.activity_type == "text":
            if not self.text:
                raise ValueError("text activities require non-blank text")
        elif self.activity_type in ("image", "video", "audio"):
            if not self.media_path:
                raise ValueError(
                    f"{self.activity_type} activities require media_path",
                )
        return self


# ---------------------------------------------------------------------------
# Analysis output (Milestone 1+)
# ---------------------------------------------------------------------------


class SentimentEvidence(BaseModel):
    """Single-modality (or overall) sentiment evidence."""

    label: SentimentLabel
    score: float = Field(
        ...,
        description="POC sentiment score approximately in [-1, +1].",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[dict[str, float]] = None
    model: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class FusionDiagnostics(BaseModel):
    """Deterministic explainable diagnostics for late fusion (no LLM)."""

    contributing_modalities: list[str] = Field(default_factory=list)
    configured_weights: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    modality_conflict: bool = False
    disagreement_score: float = 0.0
    thresholds: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    note: str = "POC evaluation defaults only; not client scoring rules."
    source_path: Optional[str] = None


class ModalityBundle(BaseModel):
    """Per-modality evidence. Unused modalities are omitted (not null-filled)."""

    text: Optional[SentimentEvidence] = None  # caption / primary text
    visual: Optional[SentimentEvidence] = None
    ocr: Optional[SentimentEvidence] = None
    speech: Optional[SentimentEvidence] = None


class SpeechSegment(BaseModel):
    """Timed ASR segment from faster-whisper."""

    start: float
    end: float
    text: str


class SpeechAnalysisResult(BaseModel):
    """Structured speech-branch output (Milestone 4). Not full video fusion."""

    transcript: Optional[str] = None
    language: Optional[str] = None
    segments: list[SpeechSegment] = Field(default_factory=list)
    transcription_seconds: Optional[float] = None
    audio_duration_seconds: Optional[float] = None
    sentiment: Optional[SentimentEvidence] = None
    asr_model: str
    warnings: list[str] = Field(default_factory=list)
    details: Optional[dict[str, Any]] = None

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PocRuntimeInfo(BaseModel):
    """Configured models and sampling settings visible in analysis output."""

    models: dict[str, str] = Field(default_factory=dict)
    video_sampling: Optional[dict[str, Any]] = None
    fusion_source: Optional[str] = None
    note: str = "POC evaluation defaults only; not client scoring rules."


class AnalysisBlock(BaseModel):
    overall: SentimentEvidence
    modalities: ModalityBundle
    fusion: Optional[FusionDiagnostics] = None
    runtime: Optional[PocRuntimeInfo] = None
    warnings: list[str] = Field(default_factory=list)
    ocr_text: Optional[str] = Field(
        default=None,
        description="Normalized OCR string when extracted (may exist without ocr sentiment).",
    )
    transcript: Optional[str] = Field(
        default=None,
        description="ASR transcript when available (speech / video).",
    )
    video: Optional[VideoDiagnostics] = Field(
        default=None,
        description="Compact video sampling / processing diagnostics.",
    )


class VideoFrameDebug(BaseModel):
    """Optional per-frame debug row (omitted from default JSON)."""

    index: int
    timestamp_seconds: Optional[float] = None
    visual_label: Optional[SentimentLabel] = None
    visual_score: Optional[float] = None
    visual_confidence: Optional[float] = None
    ocr_preview: Optional[str] = None
    error: Optional[str] = None


class VideoDiagnostics(BaseModel):
    """Compact video-level processing diagnostics (default output stays small)."""

    duration_seconds: Optional[float] = None
    sampling_strategy: str = "fixed_fps"
    sampling_fps: Optional[float] = None
    frames_extracted: int = 0
    frames_analyzed: int = 0
    frame_timestamps: list[float] = Field(default_factory=list)
    extraction_seconds: Optional[float] = None
    processing_seconds: Optional[float] = None
    has_audio: Optional[bool] = None
    scene_count: Optional[int] = None
    frame_debug: Optional[list[VideoFrameDebug]] = Field(
        default=None,
        description="Present only when debug=True on the video analyzer.",
    )


class InputMetadata(BaseModel):
    text_length: Optional[int] = None
    text_preview: Optional[str] = None
    media_path: Optional[str] = None
    created_at: Optional[datetime] = None
    content_kind: Optional[ContentKind] = None
    extra: Optional[dict[str, Any]] = None


class ActivityAnalysisResult(BaseModel):
    """Standardized activity-level sentiment result."""

    activity_id: str
    user_id: Optional[str] = None
    activity_type: ActivityType
    input: InputMetadata
    analysis: AnalysisBlock
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def model_dump_json_compatible(self) -> dict[str, Any]:
        """Serialize with ISO timestamps for CLI / file output."""
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Batch processing outcomes
# ---------------------------------------------------------------------------


class BatchRecordOutcome(BaseModel):
    """Per-record result for JSONL batch ingestion."""

    line_number: int
    status: BatchRecordStatus
    activity_id: Optional[str] = None
    user_id: Optional[str] = None
    activity_type: Optional[str] = None
    error: Optional[str] = None
    note: Optional[str] = None
    result: Optional[ActivityAnalysisResult] = None

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BatchSummary(BaseModel):
    """Aggregate metrics for a batch run."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    processed: int = 0
    unsupported: int = 0
    failed: int = 0
    skipped: int = 0


class BatchProcessingResult(BaseModel):
    """Full batch response: summary + per-record outcomes."""

    source: str
    summary: BatchSummary
    records: list[BatchRecordOutcome]
    batch_id: Optional[str] = None

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DailyUserScore(BaseModel):
    """POC daily user-level sentiment aggregate — NOT the client business score."""

    user_id: str
    score_date: str
    activity_count: int = 0
    valid_analysis_count: int = 0
    mean_sentiment_score: Optional[float] = None
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    daily_sentiment_label: Optional[SentimentLabel] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note: str = (
        "POC daily aggregate — mean of stored activity sentiment scores; "
        "NOT the future client business score"
    )

    def model_dump_json_compatible(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
