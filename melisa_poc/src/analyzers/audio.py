"""Speech / audio analysis via FFmpeg + faster-whisper (English MVP)."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Union

from src.analyzers.text import TextSentimentAnalyzer
from src.config import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_MODEL,
)
from src.media.ffmpeg_utils import FFmpegError, FFmpegNotFoundError, extract_audio_wav
from src.schemas import SentimentEvidence, SpeechAnalysisResult, SpeechSegment

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class AudioAnalyzer:
    """Extract audio (FFmpeg) and transcribe English speech (faster-whisper).

    Designed as the speech branch for future video activities. Does not perform
    frame sampling or multimodal video fusion.
    """

    def __init__(
        self,
        *,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE,
        language: str = DEFAULT_ASR_LANGUAGE,
        device: str = "cpu",
        ffmpeg_path: Optional[str] = None,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
    ) -> None:
        self.whisper_model_name = whisper_model
        self.compute_type = compute_type
        self.language = language
        self.device = device
        self.ffmpeg_path = ffmpeg_path
        self._text_analyzer = text_analyzer
        self._whisper_model: Any = None

    def set_text_analyzer(self, text_analyzer: TextSentimentAnalyzer) -> None:
        self._text_analyzer = text_analyzer

    @property
    def is_loaded(self) -> bool:
        return self._whisper_model is not None

    @staticmethod
    def validate_media_path(media_path: PathLike) -> Path:
        path = Path(media_path)
        if not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")
        return path

    def load(self) -> None:
        """Lazily load the faster-whisper model (CPU by default)."""
        if self.is_loaded:
            return

        from faster_whisper import WhisperModel

        logger.info(
            "Loading ASR model=%s device=%s compute_type=%s",
            self.whisper_model_name,
            self.device,
            self.compute_type,
        )
        self._whisper_model = WhisperModel(
            self.whisper_model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("ASR model ready: %s", self.whisper_model_name)

    def analyze(self, media_path: PathLike) -> SpeechAnalysisResult:
        """Extract audio if needed, transcribe, and optionally score transcript sentiment.

        Never fabricates transcript text. Empty / no-speech media returns warnings
        and omits speech sentiment.
        """
        source = self.validate_media_path(media_path)
        warnings: list[str] = []
        tmp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        wav_path: Optional[Path] = None

        try:
            tmp_dir = tempfile.TemporaryDirectory(prefix="myuni_asr_")
            wav_path = Path(tmp_dir.name) / "audio_16k_mono.wav"
            try:
                extract_audio_wav(
                    source,
                    wav_path,
                    ffmpeg_path=self.ffmpeg_path,
                )
            except FFmpegNotFoundError:
                raise
            except FFmpegError as exc:
                # Propagate extraction failures as clear errors (caller/batch can catch).
                raise FFmpegError(str(exc)) from exc

            self.load()
            assert self._whisper_model is not None

            started = time.perf_counter()
            segments_iter, info = self._whisper_model.transcribe(
                str(wav_path),
                language=self.language,
                vad_filter=True,
            )
            segments: list[SpeechSegment] = []
            text_parts: list[str] = []
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append(
                    SpeechSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=text,
                    ),
                )
                text_parts.append(text)
            transcription_seconds = time.perf_counter() - started

            transcript = " ".join(text_parts).strip() or None
            detected_language = getattr(info, "language", None) or self.language
            audio_duration = getattr(info, "duration", None)
            audio_duration_f = float(audio_duration) if audio_duration is not None else None

            sentiment: Optional[SentimentEvidence] = None
            if transcript is None:
                warnings.append("No speech detected (empty transcript)")
            elif self._text_analyzer is None:
                warnings.append(
                    "Transcript available but text sentiment analyzer is not configured",
                )
            else:
                try:
                    sentiment = self._text_analyzer.analyze(transcript)
                    sentiment = sentiment.model_copy(
                        update={
                            "details": {
                                **(sentiment.details or {}),
                                "source": "speech",
                                "transcript_preview": transcript[:200],
                                "asr_model": self.whisper_model_name,
                            },
                        },
                    )
                except ValueError as exc:
                    warnings.append(f"Transcript could not be scored: {exc}")

            logger.info(
                "ASR complete path=%s chars=%s segments=%s duration=%.2fs warnings=%s",
                source,
                len(transcript) if transcript else 0,
                len(segments),
                transcription_seconds,
                len(warnings),
            )

            return SpeechAnalysisResult(
                transcript=transcript,
                language=str(detected_language) if detected_language else self.language,
                segments=segments,
                transcription_seconds=float(transcription_seconds),
                audio_duration_seconds=audio_duration_f,
                sentiment=sentiment,
                asr_model=self.whisper_model_name,
                warnings=warnings,
                details={
                    "media_path": str(source),
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "language_assumed": self.language,
                },
            )
        finally:
            # Temporary WAV and directory are removed here even on errors.
            if tmp_dir is not None:
                try:
                    tmp_dir.cleanup()
                except OSError as exc:
                    logger.warning("Failed to clean temporary ASR files: %s", exc)
