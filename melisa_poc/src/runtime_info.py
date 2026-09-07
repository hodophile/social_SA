"""Helpers for exposing POC runtime configuration in analysis output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from src.config import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_FUSION,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VIDEO_MAX_FRAMES,
    DEFAULT_VIDEO_MAX_OCR_FRAMES,
    DEFAULT_VIDEO_SAMPLE_FPS,
    DEFAULT_VISUAL_MODEL,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_MODEL,
)
from src.schemas import PocRuntimeInfo

if TYPE_CHECKING:
    from src.pipeline import MyUniSentimentPipeline


def build_poc_runtime_info(
    pipeline: Optional["MyUniSentimentPipeline"] = None,
    *,
    video_sampling_strategy: Optional[str] = None,
) -> PocRuntimeInfo:
    """Build runtime metadata from configured defaults and optional live pipeline."""
    models = {
        "text": DEFAULT_TEXT_MODEL,
        "visual": DEFAULT_VISUAL_MODEL,
        "asr": DEFAULT_WHISPER_MODEL,
        "asr_compute_type": DEFAULT_WHISPER_COMPUTE_TYPE,
        "asr_language": DEFAULT_ASR_LANGUAGE,
        "fusion": "poc-fusion",
    }
    if pipeline is not None:
        models["text"] = pipeline.text_analyzer.model_name
        models["asr"] = pipeline.audio_analyzer.whisper_model_name

    strategy = video_sampling_strategy
    sampling: dict[str, Any] = {
        "fps": DEFAULT_VIDEO_SAMPLE_FPS,
        "max_frames": DEFAULT_VIDEO_MAX_FRAMES,
        "max_ocr_frames": DEFAULT_VIDEO_MAX_OCR_FRAMES,
    }
    if pipeline is not None:
        va = pipeline.video_analyzer
        strategy = strategy or va.frame_sampler.name
        sampling = {
            "strategy": strategy,
            "fps": va.sampling.fps,
            "max_frames": va.sampling.max_frames,
            "max_ocr_frames": va.sampling.max_ocr_frames,
        }
    elif strategy:
        sampling["strategy"] = strategy

    return PocRuntimeInfo(
        models=models,
        video_sampling=sampling,
        fusion_source=DEFAULT_FUSION.source_path,
        note=DEFAULT_FUSION.note,
    )
