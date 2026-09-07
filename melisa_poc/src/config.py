"""POC configuration for models, OCR, video sampling, and fusion weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


# Text sentiment (RoBERTa).
DEFAULT_TEXT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Exact Hugging Face checkpoint used for zero-shot visual concept scoring.
DEFAULT_VISUAL_MODEL = "google/siglip2-base-patch16-224"

# Zero-shot candidate prompts → sentiment labels (transparent, not a trained classifier).
DEFAULT_VISUAL_PROMPTS: dict[str, str] = {
    "positive": "a positive, joyful, pleasant or uplifting social media image",
    "neutral": "a neutral, ordinary or informational social media image",
    "negative": "a negative, sad, upsetting, unpleasant or distressing social media image",
}

# Facial-expression concept prompts (SigLIP zero-shot on face crops / face-gated frames).
DEFAULT_FACIAL_EXPRESSION_PROMPTS: dict[str, str] = {
    "positive": "a person or character with a happy smiling facial expression",
    "neutral": "a person or character with a calm neutral facial expression",
    "negative": "a person or character with a sad angry or upset facial expression",
}

# Face presence gate (positive≈face, negative≈no face, neutral≈ambiguous).
DEFAULT_FACE_GATE_PROMPTS: dict[str, str] = {
    "positive": "a close-up of a human or character face showing a facial expression",
    "neutral": "an ambiguous image that may or may not include a face",
    "negative": "a scene object or landscape with no visible face",
}

# Minimum SigLIP "face" probability to accept a Haar-miss full-frame expression score.
FACE_GATE_MIN_PROBABILITY = 0.45

# Minimum alphanumeric characters before OCR text is treated as meaningful.
OCR_MIN_ALNUM_CHARS = 3

# faster-whisper model size. base.en is CPU-friendly on ~16 GB RAM.
DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_ASR_LANGUAGE = "en"

# Video frame sampling. Baseline remains fixed ~1 FPS; scene_keyframe is optional.
DEFAULT_VIDEO_SAMPLE_FPS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 12
DEFAULT_VIDEO_MAX_OCR_FRAMES = 8
DEFAULT_VIDEO_SAMPLING_STRATEGY = "fixed_fps"

# Aliases matching the client multimodal milestone naming.
VIDEO_SAMPLE_FPS = DEFAULT_VIDEO_SAMPLE_FPS
MAX_VIDEO_FRAMES = DEFAULT_VIDEO_MAX_FRAMES

# RoBERTa max sequence length; longer transcripts are chunked (not silently truncated).
TEXT_MAX_LENGTH = 512
TEXT_CHUNK_SIZE = 480
TEXT_CHUNK_STRIDE = 64

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSION_YAML = _REPO_ROOT / "config" / "fusion.yaml"


@dataclass(frozen=True)
class VideoSamplingConfig:
    """Configurable fixed-FPS frame sampling with pathological-count safeguards."""

    fps: float = DEFAULT_VIDEO_SAMPLE_FPS
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES
    max_ocr_frames: int = DEFAULT_VIDEO_MAX_OCR_FRAMES

    def effective_fps(self, duration_seconds: float) -> float:
        """Return FPS that keeps expected frame count within ``max_frames``."""
        if duration_seconds <= 0:
            return self.fps
        expected = duration_seconds * self.fps
        if expected <= self.max_frames:
            return self.fps
        return max(self.max_frames / duration_seconds, 1e-6)


@dataclass(frozen=True)
class FusionThresholds:
    """POC label thresholds on the fused score (not scientifically validated)."""

    positive_above: float = 0.15
    negative_below: float = -0.15


@dataclass(frozen=True)
class FusionConflictConfig:
    """POC conflict detection parameters."""

    min_confidence: float = 0.40
    min_polarity: float = 0.35
    disagreement_threshold: float = 0.90
    confidence_penalty: float = 0.50


@dataclass(frozen=True)
class FusionConfig:
    """POC-only late fusion settings loaded from ``config/fusion.yaml``.

    NOT the client business scoring methodology.
    """

    modality_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "text": 1.0,
            "visual": 1.0,
            "ocr": 0.8,
            "speech": 1.0,
        },
    )
    thresholds: FusionThresholds = field(default_factory=FusionThresholds)
    conflict: FusionConflictConfig = field(default_factory=FusionConflictConfig)
    note: str = "POC evaluation defaults only; not client scoring rules."
    source_path: Optional[str] = None

    @property
    def neutral_band(self) -> float:
        """Backward-compatible half-width around zero using symmetric thresholds."""
        return max(abs(self.thresholds.positive_above), abs(self.thresholds.negative_below))


def _as_float_map(raw: Any, fallback: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(fallback)
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out or dict(fallback)


def load_fusion_config(path: Optional[Path] = None) -> FusionConfig:
    """Load fusion settings from YAML; fall back to built-in POC defaults."""
    yaml_path = Path(path) if path is not None else DEFAULT_FUSION_YAML
    defaults = FusionConfig()

    if not yaml_path.is_file():
        return FusionConfig(source_path=str(yaml_path))

    try:
        import yaml
    except ImportError:
        return FusionConfig(
            note=defaults.note + " (PyYAML missing; using built-in defaults)",
            source_path=str(yaml_path),
        )

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    weights = _as_float_map(data.get("modality_weights"), defaults.modality_weights)
    thr_raw = data.get("thresholds") or {}
    conflict_raw = data.get("conflict") or {}

    thresholds = FusionThresholds(
        positive_above=float(thr_raw.get("positive_above", defaults.thresholds.positive_above)),
        negative_below=float(thr_raw.get("negative_below", defaults.thresholds.negative_below)),
    )
    conflict = FusionConflictConfig(
        min_confidence=float(conflict_raw.get("min_confidence", defaults.conflict.min_confidence)),
        min_polarity=float(conflict_raw.get("min_polarity", defaults.conflict.min_polarity)),
        disagreement_threshold=float(
            conflict_raw.get("disagreement_threshold", defaults.conflict.disagreement_threshold),
        ),
        confidence_penalty=float(
            conflict_raw.get("confidence_penalty", defaults.conflict.confidence_penalty),
        ),
    )
    note = str(data.get("note") or defaults.note).strip()

    return FusionConfig(
        modality_weights=weights,
        thresholds=thresholds,
        conflict=conflict,
        note=note,
        source_path=str(yaml_path),
    )


DEFAULT_FUSION = load_fusion_config()
DEFAULT_VIDEO_SAMPLING = VideoSamplingConfig()
