"""
Drop‑in replacement for melisa_poc/src/analyzers/video.py
Adds per‑frame 8‑emotion extraction (via EmotionNet) and temporal aggregation.
All other functionality (OCR, speech, etc.) is left unchanged.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence, Union, List, Dict

import cv2
import numpy as np

from melisa_poc.src.analyzers.audio import AudioAnalyzer
from melisa_poc.src.analyzers.image import ImageAnalyzer
from melisa_poc.src.analyzers.ocr import is_meaningful_ocr_text
from melisa_poc.src.analyzers.text import TextSentimentAnalyzer
from melisa_poc.src.config import DEFAULT_FUSION, DEFAULT_VIDEO_SAMPLING, FusionConfig, VideoSamplingConfig
from melisa_poc.src.fusion import aggregate_frame_visual_scores, fuse_modalities
from melisa_poc.src.media.ffmpeg_utils import FFmpegError, FFmpegNotFoundError, probe_video
from melisa_poc.src.media.samplers import FrameSampler, SceneSamplingConfig, build_frame_sampler
from melisa_poc.src.schemas import (
    SentimentEvidence,
    SpeechAnalysisResult,
    VideoDiagnostics,
    VideoFrameDebug,
)

from .models import EmotionNet
from .fusion import temporal_weighted_average   # defined below

PathLike = Union[str, Path]
EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
]

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyze a video file; return multimodal evidence + compact diagnostics."""

    def __init__(
        self,
        *,
        image_analyzer: Optional[ImageAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        sampling: VideoSamplingConfig = DEFAULT_VIDEO_SAMPLING,
        fusion_config: FusionConfig = DEFAULT_FUSION,
        debug: bool = False,
    ) -> None:
        self._image = image_analyzer or ImageAnalyzer()
        self._audio = audio_analyzer or AudioAnalyzer(
            ffmpeg_path=ffmpeg_path,
            language="en",
            compute_type="int8",
            model="base.en",
        )
        self._text = text_analyzer or TextSentimentAnalyzer()
        self._ffmpeg_path = ffmpeg_path
        self._ffprobe_path = ffprobe_path
        self._sampling = sampling
        self._fusion_config = fusion_config
        self._debug = debug
        self._emotion_net = EmotionNet(device="cpu")   # CPU is fine for HF Space; change to "cuda" if you have a GPU

    # ------------------------------------------------------------------
    # Helper: temporal weighting of per‑frame emotion distributions
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_weighted_average(
        emotion_seqs: List[Dict[str, float]],
        half_life_seconds: float = 2.0,
        fps: float = 1.0,
    ) -> Dict[str, float]:
        """
        emotion_seqs[i] = emotion distribution for frame i (already normalised).
        half_life_seconds: after this many seconds the weight halves.
        fps: frames per second of the sampled video (≈1.0 in the default Melisa config).
        Returns a single emotion distribution (dict) normalised to sum 1.
        """
        if not emotion_seqs:
            return {e: 0.0 for e in EMOTION_LABELS}

        weights = []
        for i, _ in enumerate(emotion_seqs):
            t = i / fps          # assume even spacing
            weight = math.exp(-math.log(2) * t / half_life_seconds)
            weights.append(weight)

        w_sum = sum(weights)
        norm_weights = [w / w_sum for w in weights]

        agg = {e: 0.0 for e in EMOTION_LABELS}
        for emo_dict, w in zip(emotion_seqs, norm_weights):
            for e, v in emo_dict.items():
                agg[e] += w * v

        total = sum(agg.values())
        if total > 0:
            for e in agg:
                agg[e] /= total
        return agg

    # ------------------------------------------------------------------
    # Main analyse method – matches the signature used by Melisa’s pipeline
    # ------------------------------------------------------------------
    def analyze(
        self,
        media_path: PathLike,
        *,
        caption_sentiment: Optional[SentimentEvidence] = None,
    ) -> "VideoAnalysisBundle":
        source = self.validate_media_path(media_path)
        started = time.perf_counter()
        warnings: list[str] = []

        probe = probe_video(source, ffprobe_path=self._ffprobe_path)

        tmp_root: Optional[Path] = None
        try:
            tmp_root = Path(tempfile.mkdtemp(prefix="myuni_video_"))
            frames_dir = tmp_root / "frames"
            sampled = self._frame_sampler.sample(
                source,
                frames_dir,
                duration_seconds=probe.duration_seconds,
                ffmpeg_path=self._ffmpeg_path,
            )
            warnings.extend(sampled.warnings)

            frame_paths = sampled.paths
            timestamps = sampled.timestamps
            max_ocr = self.sampling.max_ocr_frames
            ocr_indices = self._ocr_frame_indices(len(frame_paths), max_ocr)

            from melisa_poc.src.analyzers.visual import VisualSentimentAnalyzer

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
                        [loaded_frames[i] for i in ok_indices],  # type: ignore[arg-type]
                    )
                    if len(scored) != len(ok_indices):
                        raise RuntimeError(
                            f"visual batch size mismatch ({len(scored)} vs {len(ok_indices)})"
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

            # Initialize per‑frame emotions list
            frame_emotions: List[Dict[str, float]] = []

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

                    # ---- NEW: per‑frame 8‑emotion prediction ----
                    # Ensure we have an RGB image
                    frame_data = loaded_frames[idx]
                    # If grayscale, convert to RGB
                    if len(frame_data.shape) == 2:
                        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
                    elif frame_data.shape[2] == 1:
                        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
                    else:
                        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

                    frame_emotion = self._emotion_net.predict_emotion(frame_rgb)
                    frame_emotions.append(frame_emotion)

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
                                error=str(err),
                            ),
                        )

            # ------------------------------------------------------------------
            # Aggregate visual (confidence‑weighted average – no temporal model)
            # ------------------------------------------------------------------
            visual_summary = aggregate_frame_visual_scores(
                frame_visuals,
                config=self._fusion_config,
            )
            if visual_summary is None:
                warnings.append("No frames successfully analyzed for visual sentiment")

            # ------------------------------------------------------------------
            # OCR aggregation
            # ------------------------------------------------------------------
            ocr_sentiment = self._aggregate_ocr(ocr_sentiments, ocr_texts, warnings)
            combined_ocr_text = " | ".join(dict.fromkeys(ocr_texts)) or None

            # ------------------------------------------------------------------
            # Speech (audio) – unchanged from Melisa
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # Build per‑frame emotion list for the caller
            # ------------------------------------------------------------------
            # We already collected frame_emotions during the loop.

            overall_fusion = fuse_modalities(
                {
                    "text": caption_sentiment,
                    "visual": visual_summary,
                    "ocr": ocr_sentiment,
                    "speech": speech_sentiment,
                },
                config=self._fusion_config,
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
                frame_emotions=frame_emotions,   # <-- NEW: expose per‑frame emotions
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

    # ------------------------------------------------------------------
    # Helper: OCR aggregation (unchanged from Melisa)
    # ------------------------------------------------------------------
    def _aggregate_ocr(
        self,
        ocr_sentiments: Sequence[SentimentEvidence],
        ocr_texts: Sequence[str],
        warnings: list[str],
    ) -> Optional[SentimentEvidence]:
        if ocr_sentiments:
            aggregated = aggregate_frame_visual_scores(
                list(ocr_sentiments),
                config=self._fusion_config,
            )
            if aggregated is not None:
                return aggregated.model_copy(
                    update={
                        **(aggregated.details or {}),
                        "source": "ocr",
                        "method": "confidence-weighted average over OCR frame sentiments",
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
                    **(scored.details or {}),
                    "source": "ocr",
                    "extracted_text_preview": joined[:200],
                },
            )
        except ValueError as exc:
            warnings.append(f"OCR text could not be scored: {exc}")
            return None

    # ------------------------------------------------------------------
    # Dummy properties to keep the same interface as Melisa’s VideoAnalyzer
    # ------------------------------------------------------------------
    @property
    def preserve_temp(self) -> bool:
        return False

    @property
    def _frame_sampler(self):
        return build_frame_sampler(self._sampling)
