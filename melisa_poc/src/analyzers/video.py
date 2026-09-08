"""Video activity analysis: pluggable frame sampling + OCR subset + speech + caption."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from src.analyzers.audio import AudioAnalyzer
from src.analyzers.image import ImageAnalyzer
from src.analyzers.ocr import is_meaningful_ocr_text
from src.analyzers.text import TextSentimentAnalyzer
from src.config import DEFAULT_FUSION, DEFAULT_VIDEO_SAMPLING, FusionConfig, VideoSamplingConfig
from src.fusion import aggregate_frame_visual_scores, fuse_modalities
from src.media.ffmpeg_utils import FFmpegError, FFmpegNotFoundError, probe_video
from src.media.samplers import FrameSampler, SceneSamplingConfig, build_frame_sampler
from src.schemas import (
    SentimentEvidence,
    SpeechAnalysisResult,
    VideoDiagnostics,
    VideoFrameDebug,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class VideoAnalysisBundle:
    """Pre-activity multimodal evidence for one video (pipeline wraps into ActivityAnalysisResult)."""

    visual: Optional[SentimentEvidence]
    ocr: Optional[SentimentEvidence]
    ocr_text: Optional[str]
    speech: Optional[SentimentEvidence]
    transcript: Optional[str]
    speech_result: Optional[SpeechAnalysisResult]
    diagnostics: VideoDiagnostics
    warnings: list[str] = field(default_factory=list)
    overall: Optional[SentimentEvidence] = None


def _ocr_frame_indices(n_frames: int, max_ocr: int) -> set[int]:
    if n_frames <= 0 or max_ocr <= 0:
        return set()
    if n_frames <= max_ocr:
        return set(range(n_frames))
    positions = {
        int(round(i * (n_frames - 1) / (max_ocr - 1)))
        for i in range(max_ocr)
    }
    return positions


class VideoAnalyzer:
    """Video analyzer with pluggable frame sampling strategies.

    Baseline strategy: ``fixed_fps`` (~1 FPS via FFmpeg).
    Alternative: ``scene_keyframe`` (PySceneDetect + FFmpeg stills).
    """

    def __init__(
        self,
        *,
        image_analyzer: Optional[ImageAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
        sampling: VideoSamplingConfig = DEFAULT_VIDEO_SAMPLING,
        scene_sampling: Optional[SceneSamplingConfig] = None,
        frame_sampler: Optional[FrameSampler] = None,
        sampling_strategy: str = "fixed_fps",
        fusion_config: FusionConfig = DEFAULT_FUSION,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        debug: bool = False,
        preserve_temp: bool = False,
    ) -> None:
        self._image = image_analyzer or ImageAnalyzer(text_analyzer=text_analyzer)
        self._audio = audio_analyzer or AudioAnalyzer(text_analyzer=text_analyzer)
        self._text = text_analyzer
        self.sampling = sampling
        self.scene_sampling = scene_sampling or SceneSamplingConfig(
            max_frames=sampling.max_frames,
            max_ocr_frames=sampling.max_ocr_frames,
        )
        self.fusion_config = fusion_config
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.debug = debug
        self.preserve_temp = preserve_temp
        self._frame_sampler = frame_sampler or build_frame_sampler(
            sampling_strategy,
            sampling=self.sampling,
            scene=self.scene_sampling,
        )

    @property
    def frame_sampler(self) -> FrameSampler:
        return self._frame_sampler

    def set_frame_sampler(self, sampler: FrameSampler) -> None:
        self._frame_sampler = sampler

    def set_sampling_strategy(self, strategy: str) -> None:
        self._frame_sampler = build_frame_sampler(
            strategy,
            sampling=self.sampling,
            scene=self.scene_sampling,
        )

    def set_text_analyzer(self, text_analyzer: TextSentimentAnalyzer) -> None:
        self._text = text_analyzer
        self._image.set_text_analyzer(text_analyzer)
        self._audio.set_text_analyzer(text_analyzer)

    @staticmethod
    def validate_media_path(media_path: PathLike) -> Path:
        path = Path(media_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path

    def analyze(
        self,
        media_path: PathLike,
        *,
        caption_sentiment: Optional[SentimentEvidence] = None,
    ) -> VideoAnalysisBundle:
        """Analyze a video file; return multimodal evidence + compact diagnostics."""
        source = self.validate_media_path(media_path)
        started = time.perf_counter()
        warnings: list[str] = []

        probe = probe_video(source, ffprobe_path=self.ffprobe_path)

        tmp_root: Optional[Path] = None
        try:
            tmp_root = Path(tempfile.mkdtemp(prefix="myuni_video_"))
            frames_dir = tmp_root / "frames"
            sampled = self._frame_sampler.sample(
                source,
                frames_dir,
                duration_seconds=probe.duration_seconds,
                ffmpeg_path=self.ffmpeg_path,
            )
            warnings.extend(sampled.warnings)

            frame_paths = sampled.paths
            timestamps = sampled.timestamps
            max_ocr = self.sampling.max_ocr_frames
            ocr_indices = _ocr_frame_indices(len(frame_paths), max_ocr)

            from src.analyzers.visual import VisualSentimentAnalyzer

            loaded_frames: list[object] = [None] * len(frame_paths)
            frame_errors: dict[int, str] = {}
            for idx, frame_path in enumerate(frame_paths):
                try:
                    loaded_frames[idx] = VisualSentimentAnalyzer.load_image(frame_path)
                except Exception as exc:  # noqa: BLE001
                    frame_errors[idx] = str(exc)

            ok_indices = [i for i, img in enumerate(loaded_frames) if img is not None]
            visual_by_idx: dict[int, SentimentEvidence] = {}
            if ok_indices:
                try:
                    scored = self._image._visual.analyze_images(
                        [loaded_frames[i] for i in ok_indices],  # type: ignore[list-item]
                    )
                    if len(scored) != len(ok_indices):
                        raise RuntimeError(
                            f"visual batch size mismatch ({len(scored)} vs {len(ok_indices)})",
                        )
                    visual_by_idx = dict(zip(ok_indices, scored))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Batch visual scoring failed: %s", exc)
                    for idx in ok_indices:
                        frame_errors[idx] = str(exc)

            frame_visuals: list[SentimentEvidence] = []
            ocr_sentiments: list[SentimentEvidence] = []
            ocr_texts: list[str] = []
            frame_debug: list[VideoFrameDebug] = []
            frames_analyzed = 0
            ocr_unavailable_noted = False

            for idx, frame_path in enumerate(frame_paths):
                ts = timestamps[idx] if idx < len(timestamps) else None
                visual = visual_by_idx.get(idx)
                if visual is None:
                    err = frame_errors.get(idx, "no visual evidence")
                    msg = f"frame[{idx}] analysis failed: {err}"
                    warnings.append(msg)
                    logger.warning("%s", msg)
                    if self.debug:
                        frame_debug.append(
                            VideoFrameDebug(
                                index=idx,
                                timestamp_seconds=ts,
                                error=str(err),
                            ),
                        )
                    continue
                try:
                    ocr_text = None
                    ocr_sentiment = None
                    frame_warnings: list[str] = []
                    if idx in ocr_indices:
                        ocr_text, ocr_sentiment, frame_warnings = (
                            self._image.extract_ocr_evidence(loaded_frames[idx])  # type: ignore[arg-type]
                        )

                    frame_visuals.append(visual)
                    frames_analyzed += 1

                    for w in frame_warnings:
                        if "OCR unavailable" in w:
                            if not ocr_unavailable_noted:
                                warnings.append(w)
                                ocr_unavailable_noted = True
                        elif "OCR returned no text" not in w and w not in warnings:
                            warnings.append(f"frame[{idx}]: {w}")

                    if ocr_text and is_meaningful_ocr_text(ocr_text):
                        ocr_texts.append(ocr_text)
                    if ocr_sentiment is not None:
                        ocr_sentiments.append(ocr_sentiment)

                    if self.debug:
                        frame_debug.append(
                            VideoFrameDebug(
                                index=idx,
                                timestamp_seconds=ts,
                                visual_label=visual.label,
                                visual_score=visual.score,
                                visual_confidence=visual.confidence,
                                ocr_preview=(ocr_text or "")[:80] or None,
                            ),
                        )
                except Exception as exc:  # noqa: BLE001
                    msg = f"frame[{idx}] analysis failed: {exc}"
                    warnings.append(msg)
                    logger.warning("%s", msg)
                    if self.debug:
                        frame_debug.append(
                            VideoFrameDebug(
                                index=idx,
                                timestamp_seconds=ts,
                                error=str(exc),
                            ),
                        )

            visual_summary = aggregate_frame_visual_scores(
                frame_visuals,
                config=self.fusion_config,
            )
            if visual_summary is None:
                warnings.append("No frames successfully analyzed for visual sentiment")

            ocr_sentiment = self._aggregate_ocr(ocr_sentiments, ocr_texts, warnings)
            combined_ocr_text = " | ".join(dict.fromkeys(ocr_texts)) or None

            speech_sentiment: Optional[SentimentEvidence] = None
            transcript: Optional[str] = None
            speech_result: Optional[SpeechAnalysisResult] = None

            if not probe.has_audio:
                warnings.append("Video has no audio stream; speech modality skipped")
            else:
                try:
                    speech_result = self._audio.analyze(source)
                    transcript = speech_result.transcript
                    speech_sentiment = speech_result.sentiment
                    for w in speech_result.warnings:
                        if w not in warnings:
                            warnings.append(w)
                except FFmpegNotFoundError:
                    raise
                except FFmpegError as exc:
                    warnings.append(f"Speech/audio extraction failed: {exc}")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Speech analysis failed: {exc}")
                    logger.exception("Speech analysis failed for %s", source)

            processing_seconds = time.perf_counter() - started
            diagnostics = VideoDiagnostics(
                duration_seconds=probe.duration_seconds,
                sampling_strategy=sampled.strategy,
                sampling_fps=sampled.sampling_fps,
                frames_extracted=len(frame_paths),
                frames_analyzed=frames_analyzed,
                frame_timestamps=timestamps,
                extraction_seconds=sampled.extraction_seconds,
                processing_seconds=float(processing_seconds),
                has_audio=probe.has_audio,
                scene_count=sampled.scene_count,
                frame_debug=frame_debug if self.debug else None,
            )

            overall_fusion = fuse_modalities(
                {
                    "text": caption_sentiment,
                    "visual": visual_summary,
                    "ocr": ocr_sentiment,
                    "speech": speech_sentiment,
                },
                config=self.fusion_config,
            )
            overall = overall_fusion.overall

            logger.info(
                "Video analysis complete path=%s strategy=%s frames=%s/%s overall=%s",
                source,
                sampled.strategy,
                frames_analyzed,
                len(frame_paths),
                overall.label,
            )

            return VideoAnalysisBundle(
                visual=visual_summary,
                ocr=ocr_sentiment,
                ocr_text=combined_ocr_text,
                speech=speech_sentiment,
                transcript=transcript,
                speech_result=speech_result,
                diagnostics=diagnostics,
                warnings=warnings,
                overall=overall,
            )
        finally:
            if tmp_root is not None:
                if self.preserve_temp:
                    logger.info(
                        "Preserving temporary video artifacts at %s (preserve_temp=True)",
                        tmp_root,
                    )
                else:
                    import shutil

                    try:
                        shutil.rmtree(tmp_root, ignore_errors=False)
                    except OSError as exc:
                        logger.warning("Failed to clean temporary video files: %s", exc)

    def _aggregate_ocr(
        self,
        ocr_sentiments: Sequence[SentimentEvidence],
        ocr_texts: Sequence[str],
        warnings: list[str],
    ) -> Optional[SentimentEvidence]:
        if ocr_sentiments:
            aggregated = aggregate_frame_visual_scores(
                list(ocr_sentiments),
                config=self.fusion_config,
            )
            if aggregated is not None:
                return aggregated.model_copy(
                    update={
                        "details": {
                            **(aggregated.details or {}),
                            "source": "ocr",
                            "method": "confidence-weighted average over OCR frame sentiments",
                        },
                    },
                )

        joined = " ".join(dict.fromkeys(ocr_texts)).strip()
        if not joined or not is_meaningful_ocr_text(joined):
            return None
        if self._text is None:
            warnings.append("OCR text found but text sentiment analyzer is not configured")
            return None
        try:
            scored = self._text.analyze(joined)
            return scored.model_copy(
                update={
                    "details": {
                        **(scored.details or {}),
                        "source": "ocr",
                        "extracted_text_preview": joined[:200],
                    },
                },
            )
        except ValueError as exc:
            warnings.append(f"OCR text could not be scored: {exc}")
            return None
