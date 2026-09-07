"""Video frame sampling strategies (fixed FPS baseline + scene/keyframe)."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from src.config import DEFAULT_VIDEO_SAMPLE_FPS, DEFAULT_VIDEO_MAX_FRAMES, VideoSamplingConfig
from src.media.ffmpeg_utils import (
    FFmpegError,
    FFmpegNotFoundError,
    extract_frame_at_timestamp,
    extract_frames_at_fps,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class FrameSample:
    """One extracted frame with timestamp metadata."""

    path: Path
    timestamp_seconds: float
    index: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampledFrames:
    """Common sampler output consumed by VideoAnalyzer."""

    frames: list[FrameSample]
    strategy: str
    extraction_seconds: float
    sampling_fps: Optional[float] = None
    scene_count: Optional[int] = None
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> list[Path]:
        return [f.path for f in self.frames]

    @property
    def timestamps(self) -> list[float]:
        return [f.timestamp_seconds for f in self.frames]


class FrameSampler(ABC):
    """Strategy interface for selecting representative video frames."""

    name: str

    @abstractmethod
    def sample(
        self,
        media_path: PathLike,
        output_dir: PathLike,
        *,
        duration_seconds: float,
        ffmpeg_path: Optional[str] = None,
    ) -> SampledFrames:
        """Extract frames into ``output_dir`` and return a common structure."""


class FixedFPSSampler(FrameSampler):
    """Baseline ~1 FPS (configurable) sampling via FFmpeg."""

    name = "fixed_fps"

    def __init__(self, config: Optional[VideoSamplingConfig] = None) -> None:
        self.config = config or VideoSamplingConfig()

    def sample(
        self,
        media_path: PathLike,
        output_dir: PathLike,
        *,
        duration_seconds: float,
        ffmpeg_path: Optional[str] = None,
    ) -> SampledFrames:
        warnings: list[str] = []
        effective_fps = self.config.effective_fps(duration_seconds)
        if effective_fps + 1e-9 < self.config.fps:
            warnings.append(
                f"Reduced sampling FPS from {self.config.fps} to {effective_fps:.4f} "
                f"to respect max_frames={self.config.max_frames}",
            )

        started = time.perf_counter()
        paths = extract_frames_at_fps(
            media_path,
            output_dir,
            fps=effective_fps,
            ffmpeg_path=ffmpeg_path,
        )
        if len(paths) > self.config.max_frames:
            warnings.append(
                f"Truncated extracted frames from {len(paths)} "
                f"to max_frames={self.config.max_frames}",
            )
            paths = paths[: self.config.max_frames]

        frames = [
            FrameSample(
                path=p,
                timestamp_seconds=round(i / effective_fps, 3),
                index=i,
                meta={"strategy": self.name},
            )
            for i, p in enumerate(paths)
        ]
        elapsed = time.perf_counter() - started
        return SampledFrames(
            frames=frames,
            strategy=self.name,
            extraction_seconds=float(elapsed),
            sampling_fps=float(effective_fps),
            warnings=warnings,
            details={"requested_fps": self.config.fps, "max_frames": self.config.max_frames},
        )


@dataclass(frozen=True)
class SceneSamplingConfig:
    """Configuration for PySceneDetect-based keyframe sampling."""

    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES
    max_ocr_frames: int = 8
    frames_per_scene: int = 1  # mid-scene representative by default
    content_threshold: float = 27.0
    min_scene_len: int = 15
    fallback_to_fixed_fps: bool = True
    fallback_fps: float = DEFAULT_VIDEO_SAMPLE_FPS


class SceneKeyframeSampler(FrameSampler):
    """Scene/shot detection via PySceneDetect + FFmpeg frame extraction.

    Detects scenes with ContentDetector, then extracts a small number of
    representative frames per scene (default: mid-frame). Falls back to
    FixedFPSSampler when configured, otherwise raises a clear error.
    """

    name = "scene_keyframe"

    def __init__(self, config: Optional[SceneSamplingConfig] = None) -> None:
        self.config = config or SceneSamplingConfig()

    def sample(
        self,
        media_path: PathLike,
        output_dir: PathLike,
        *,
        duration_seconds: float,
        ffmpeg_path: Optional[str] = None,
    ) -> SampledFrames:
        source = Path(media_path)
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        started = time.perf_counter()

        try:
            timestamps = self._detect_keyframe_timestamps(source)
        except Exception as exc:  # noqa: BLE001 — may fall back
            msg = f"Scene detection failed: {exc}"
            logger.warning("%s", msg)
            if self.config.fallback_to_fixed_fps:
                warnings.append(msg)
                warnings.append("Falling back to fixed_fps sampler")
                fallback = FixedFPSSampler(
                    VideoSamplingConfig(
                        fps=self.config.fallback_fps,
                        max_frames=self.config.max_frames,
                        max_ocr_frames=self.config.max_ocr_frames,
                    ),
                )
                result = fallback.sample(
                    source,
                    dest,
                    duration_seconds=duration_seconds,
                    ffmpeg_path=ffmpeg_path,
                )
                result.warnings = warnings + list(result.warnings)
                result.details = {
                    **result.details,
                    "requested_strategy": self.name,
                    "fallback_used": True,
                }
                # Keep strategy label as scene_keyframe_fallback for clarity in comparisons.
                return SampledFrames(
                    frames=result.frames,
                    strategy="scene_keyframe_fallback_fixed_fps",
                    extraction_seconds=result.extraction_seconds,
                    sampling_fps=result.sampling_fps,
                    scene_count=0,
                    warnings=result.warnings,
                    details=result.details,
                )
            raise RuntimeError(
                f"{msg}. Install PySceneDetect (`pip install scenedetect[opencv]`) "
                "or enable fallback_to_fixed_fps.",
            ) from exc

        if not timestamps:
            warnings.append("Scene detection returned no keyframe timestamps")
            if self.config.fallback_to_fixed_fps:
                warnings.append("Falling back to fixed_fps sampler")
                fallback = FixedFPSSampler(
                    VideoSamplingConfig(
                        fps=self.config.fallback_fps,
                        max_frames=self.config.max_frames,
                    ),
                )
                result = fallback.sample(
                    source,
                    dest,
                    duration_seconds=duration_seconds,
                    ffmpeg_path=ffmpeg_path,
                )
                return SampledFrames(
                    frames=result.frames,
                    strategy="scene_keyframe_fallback_fixed_fps",
                    extraction_seconds=result.extraction_seconds,
                    sampling_fps=result.sampling_fps,
                    scene_count=0,
                    warnings=warnings + list(result.warnings),
                    details={**result.details, "fallback_used": True},
                )
            raise RuntimeError("Scene detection produced no frames and fallback is disabled")

        keyframes_requested = len(timestamps)
        if len(timestamps) > self.config.max_frames:
            warnings.append(
                f"Truncated scene keyframes from {len(timestamps)} "
                f"to max_frames={self.config.max_frames}",
            )
            # Evenly keep max_frames timestamps including ends when possible.
            n = self.config.max_frames
            if n == 1:
                timestamps = [timestamps[len(timestamps) // 2]]
            else:
                idxs = sorted(
                    {
                        int(round(i * (len(timestamps) - 1) / (n - 1)))
                        for i in range(n)
                    },
                )
                timestamps = [timestamps[i] for i in idxs]

        frames: list[FrameSample] = []
        for i, ts in enumerate(timestamps):
            out_path = dest / f"scene_frame_{i:05d}.jpg"
            try:
                extract_frame_at_timestamp(
                    source,
                    out_path,
                    timestamp_seconds=ts,
                    ffmpeg_path=ffmpeg_path,
                )
                frames.append(
                    FrameSample(
                        path=out_path,
                        timestamp_seconds=round(float(ts), 3),
                        index=i,
                        meta={"strategy": self.name},
                    ),
                )
            except (FFmpegError, FFmpegNotFoundError) as exc:
                warnings.append(f"Failed extracting frame at t={ts:.3f}s: {exc}")
                if isinstance(exc, FFmpegNotFoundError):
                    raise

        if not frames:
            raise RuntimeError("Scene keyframe sampling extracted no usable frames")

        elapsed = time.perf_counter() - started
        return SampledFrames(
            frames=frames,
            strategy=self.name,
            extraction_seconds=float(elapsed),
            sampling_fps=None,
            scene_count=int(getattr(self, "_last_scene_count", 0) or 0),
            warnings=warnings,
            details={
                "content_threshold": self.config.content_threshold,
                "frames_per_scene": self.config.frames_per_scene,
                "max_frames": self.config.max_frames,
                "keyframes_requested": keyframes_requested,
            },
        )

    def _detect_keyframe_timestamps(self, source: Path) -> list[float]:
        try:
            from scenedetect import ContentDetector, detect
        except ImportError as exc:
            raise ImportError(
                "PySceneDetect is required for scene_keyframe sampling. "
                "Install with: pip install 'scenedetect[opencv]'",
            ) from exc

        scenes = detect(
            str(source),
            ContentDetector(
                threshold=self.config.content_threshold,
                min_scene_len=self.config.min_scene_len,
            ),
            show_progress=False,
            start_in_scene=True,
        )
        if not scenes:
            self._last_scene_count = 0
            return []
        self._last_scene_count = len(scenes)
        return keyframe_timestamps_from_scenes(
            scenes,
            frames_per_scene=self.config.frames_per_scene,
        )


def keyframe_timestamps_from_scenes(
    scenes: list[tuple[Any, Any]],
    *,
    frames_per_scene: int = 1,
) -> list[float]:
    """Convert PySceneDetect (start, end) pairs into representative timestamps."""
    timestamps: list[float] = []
    for start, end in scenes:
        start_s = float(start.get_seconds())
        end_s = float(end.get_seconds())
        if frames_per_scene <= 1:
            timestamps.append(start_s + max(0.0, (end_s - start_s) / 2.0))
        else:
            n = frames_per_scene
            for i in range(n):
                frac = 0.5 if n == 1 else i / (n - 1)
                timestamps.append(start_s + frac * max(0.0, end_s - start_s))
    return timestamps


def build_frame_sampler(
    strategy: str = "fixed_fps",
    *,
    sampling: Optional[VideoSamplingConfig] = None,
    scene: Optional[SceneSamplingConfig] = None,
) -> FrameSampler:
    """Factory for video frame sampling strategies."""
    key = strategy.strip().lower().replace("-", "_")
    if key in {"fixed_fps", "fixed", "fps"}:
        return FixedFPSSampler(sampling)
    if key in {"scene_keyframe", "scene", "keyframe", "scenedetect"}:
        return SceneKeyframeSampler(scene)
    raise ValueError(
        f"Unknown video sampling strategy: {strategy!r}. "
        "Use 'fixed_fps' (baseline) or 'scene_keyframe'.",
    )
