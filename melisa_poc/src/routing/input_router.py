"""Automatic input-type detection and capability-aware routing.

The Streamlit client uses this path so modality selection is never manual.
Existing CLI / batch ``analyze_activity`` paths are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_TEXT_MODEL, DEFAULT_VISUAL_MODEL, DEFAULT_WHISPER_MODEL


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class CapabilityStatus(str, Enum):
    OK = "ok"
    NOT_IMPLEMENTED = "not_implemented"
    VALIDATION_ERROR = "validation_error"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Client multimodal baseline: text, image, audio, and video analyzers.
CLIENT_ENABLED_MODALITIES: frozenset[InputType] = frozenset(
    {InputType.TEXT, InputType.IMAGE, InputType.AUDIO, InputType.VIDEO},
)

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".flac"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv"})

_IMAGE_MIME_PREFIXES = ("image/",)
_AUDIO_MIME_PREFIXES = ("audio/",)
_VIDEO_MIME_PREFIXES = ("video/",)

_IMAGE_MIME_EXACT = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/x-ms-bmp",
    },
)
_AUDIO_MIME_EXACT = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/ogg",
        "audio/flac",
        "audio/x-flac",
    },
)
_VIDEO_MIME_EXACT = frozenset(
    {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
        "video/x-matroska",
        "video/avi",
        "application/mp4",
    },
)

_INSUFFICIENT_MESSAGES = {
    InputType.IMAGE: (
        "Image detected and routed successfully, but no usable visual sentiment "
        "evidence could be produced for this file."
    ),
    InputType.AUDIO: (
        "Audio detected and routed successfully, but no usable speech transcript "
        "was produced (no-speech or insufficient text). Sentiment was not invented."
    ),
    InputType.VIDEO: (
        "Video detected and routed successfully, but neither visual nor speech "
        "evidence was usable. Insufficient evidence for a POC overall sentiment."
    ),
}


@dataclass(frozen=True)
class DetectedInput:
    """Normalized detection result before capability checks."""

    input_type: InputType
    text: Optional[str] = None
    media_path: Optional[str] = None
    mime_type: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class RoutedAnalysisResult:
    """Unified client analyze response (implemented or capability-gated)."""

    status: CapabilityStatus
    detected_input: Optional[InputType]
    message: Optional[str] = None
    analysis: object = None  # ActivityAnalysisResult when status == ok
    model_display_name: Optional[str] = None
    model_id: Optional[str] = None


class InputRouter:
    """Detect modality from text and/or uploaded media signals."""

    @staticmethod
    def _normalize_mime(mime_type: Optional[str]) -> Optional[str]:
        if mime_type is None:
            return None
        cleaned = mime_type.strip().lower()
        return cleaned or None

    @staticmethod
    def _extension(filename: Optional[str], media_path: Optional[str]) -> Optional[str]:
        for candidate in (filename, media_path):
            if not candidate:
                continue
            suffix = Path(candidate).suffix.lower()
            if suffix:
                return suffix
        return None

    @classmethod
    def classify_media(
        cls,
        *,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
        media_path: Optional[str] = None,
    ) -> Optional[InputType]:
        """Return IMAGE/AUDIO/VIDEO from MIME (primary) and extension (secondary)."""
        mime = cls._normalize_mime(mime_type)
        ext = cls._extension(filename, media_path)

        mime_kind: Optional[InputType] = None
        if mime:
            if mime in _IMAGE_MIME_EXACT or mime.startswith(_IMAGE_MIME_PREFIXES):
                mime_kind = InputType.IMAGE
            elif mime in _AUDIO_MIME_EXACT or mime.startswith(_AUDIO_MIME_PREFIXES):
                mime_kind = InputType.AUDIO
            elif mime in _VIDEO_MIME_EXACT or mime.startswith(_VIDEO_MIME_PREFIXES):
                mime_kind = InputType.VIDEO

        ext_kind: Optional[InputType] = None
        if ext in _IMAGE_EXTENSIONS:
            ext_kind = InputType.IMAGE
        elif ext in _AUDIO_EXTENSIONS:
            ext_kind = InputType.AUDIO
        elif ext in _VIDEO_EXTENSIONS:
            ext_kind = InputType.VIDEO

        if mime_kind and ext_kind and mime_kind != ext_kind:
            raise ValueError(
                f"File type mismatch: MIME suggests {mime_kind.value}, "
                f"but extension suggests {ext_kind.value}. "
                "Please upload a supported image, audio, or video file.",
            )

        return mime_kind or ext_kind

    @classmethod
    def detect(
        cls,
        *,
        text: Optional[str] = None,
        media_path: Optional[str] = None,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> DetectedInput:
        """Choose a single active input.

        Priority: non-blank text wins (media ignored). Otherwise media is required.
        """
        cleaned_text = text.strip() if isinstance(text, str) else None
        if cleaned_text:
            return DetectedInput(input_type=InputType.TEXT, text=cleaned_text)

        has_media = bool(media_path) or bool(filename) or bool(mime_type)
        if not has_media:
            raise ValueError(
                "Enter text or upload a supported media file to analyze.",
            )

        kind = cls.classify_media(
            mime_type=mime_type,
            filename=filename,
            media_path=media_path,
        )
        if kind is None:
            label = filename or media_path or mime_type or "upload"
            raise ValueError(
                f"Unsupported file type ({label}). "
                "Supported images: JPG, JPEG, PNG, WEBP. "
                "Supported audio: WAV, MP3, M4A, OGG, FLAC. "
                "Supported videos: MP4, MOV, AVI, WEBM, MKV.",
            )

        return DetectedInput(
            input_type=kind,
            media_path=media_path,
            mime_type=cls._normalize_mime(mime_type),
            filename=filename,
        )

    @classmethod
    def is_enabled(
        cls,
        input_type: InputType,
        enabled: Optional[frozenset[InputType]] = None,
    ) -> bool:
        active = CLIENT_ENABLED_MODALITIES if enabled is None else enabled
        return input_type in active

    @classmethod
    def not_implemented_message(cls, input_type: InputType) -> str:
        return (
            f"{input_type.value.title()} analysis is not enabled in this build."
        )

    @classmethod
    def insufficient_evidence_message(cls, input_type: InputType) -> str:
        return _INSUFFICIENT_MESSAGES.get(
            input_type,
            f"Insufficient evidence to score this {input_type.value} input.",
        )

    @classmethod
    def text_model_meta(cls) -> tuple[str, str]:
        """Return (display name, exact checkpoint id) for client results."""
        return "Twitter-RoBERTa", DEFAULT_TEXT_MODEL

    @classmethod
    def visual_model_meta(cls) -> tuple[str, str]:
        """Return (display name, exact checkpoint id) for visual sentiment."""
        return "SigLIP 2", DEFAULT_VISUAL_MODEL

    @classmethod
    def audio_model_meta(cls) -> tuple[str, str]:
        """Return (display name, ASR id) for audio results."""
        return "Faster-Whisper + Twitter-RoBERTa", DEFAULT_WHISPER_MODEL

    @classmethod
    def video_model_meta(cls) -> tuple[str, str]:
        """Return (display name, primary visual checkpoint) for video results."""
        return "SigLIP 2 + Faster-Whisper + Twitter-RoBERTa", DEFAULT_VISUAL_MODEL
