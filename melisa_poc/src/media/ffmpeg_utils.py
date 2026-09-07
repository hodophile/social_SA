"""FFmpeg / ffprobe helpers for audio extraction and video frame sampling."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

FFMPEG_MISSING_MESSAGE = (
    "FFmpeg executable not found on PATH. "
    "Install FFmpeg for Windows (e.g. https://www.gyan.dev/ffmpeg/builds/ "
    "or `winget install Gyan.FFmpeg`) and reopen the terminal so `ffmpeg` is available. "
    "Verify with: ffmpeg -version"
)


class FFmpegNotFoundError(RuntimeError):
    """Raised when the FFmpeg binary cannot be located."""


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg invocation fails."""


@dataclass(frozen=True)
class VideoProbeInfo:
    """Subset of ffprobe metadata used by the video analyzer."""

    duration_seconds: float
    has_video: bool
    has_audio: bool
    width: Optional[int] = None
    height: Optional[int] = None


def find_ffmpeg(ffmpeg_path: Optional[str] = None) -> str:
    """Return the FFmpeg executable path or raise an actionable error."""
    if ffmpeg_path:
        candidate = Path(ffmpeg_path)
        if candidate.is_file():
            return str(candidate)
        raise FFmpegNotFoundError(
            f"Configured FFmpeg path does not exist: {ffmpeg_path}. {FFMPEG_MISSING_MESSAGE}",
        )

    found = shutil.which("ffmpeg")
    if not found:
        raise FFmpegNotFoundError(FFMPEG_MISSING_MESSAGE)
    return found


def find_ffprobe(ffprobe_path: Optional[str] = None) -> str:
    """Return the ffprobe executable path (sibling of ffmpeg when possible)."""
    if ffprobe_path:
        candidate = Path(ffprobe_path)
        if candidate.is_file():
            return str(candidate)
        raise FFmpegNotFoundError(f"Configured ffprobe path does not exist: {ffprobe_path}")

    found = shutil.which("ffprobe")
    if found:
        return found

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix else "ffprobe")
        if sibling.is_file():
            return str(sibling)

    raise FFmpegNotFoundError(
        "ffprobe executable not found on PATH. "
        "Install a full FFmpeg build that includes ffprobe "
        "(e.g. winget install Gyan.FFmpeg). Verify with: ffprobe -version",
    )


def extract_audio_wav(
    media_path: PathLike,
    output_wav: PathLike,
    *,
    ffmpeg_path: Optional[str] = None,
    sample_rate: int = 16000,
    timeout_seconds: int = 300,
) -> Path:
    """Extract / normalize media audio to mono 16 kHz PCM WAV via FFmpeg."""
    source = Path(media_path)
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")

    dest = Path(output_wav)
    dest.parent.mkdir(parents=True, exist_ok=True)

    exe = find_ffmpeg(ffmpeg_path)
    cmd = [
        exe,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    logger.info("Extracting audio with FFmpeg: %s -> %s", source, dest)
    _run_ffmpeg(cmd, timeout_seconds=timeout_seconds, source=source)

    if not dest.is_file() or dest.stat().st_size == 0:
        raise FFmpegError(
            f"FFmpeg produced no usable audio output for {source} "
            "(file may have no audio stream)",
        )

    return dest


def probe_video(media_path: PathLike, *, ffprobe_path: Optional[str] = None) -> VideoProbeInfo:
    """Probe duration and stream presence with ffprobe."""
    source = Path(media_path)
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")

    exe = find_ffprobe(ffprobe_path)
    cmd = [
        exe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffprobe timed out while probing {source}") from exc
    except OSError as exc:
        raise FFmpegError(f"Failed to launch ffprobe: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-800:] or "no stderr"
        raise FFmpegError(f"ffprobe failed for {source}: {detail}")

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {source}") from exc

    duration_raw = (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    has_video = False
    has_audio = False
    width = height = None
    for stream in payload.get("streams") or []:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            has_video = True
            if width is None:
                width = stream.get("width")
                height = stream.get("height")
        elif codec_type == "audio":
            has_audio = True

    if not has_video:
        raise FFmpegError(f"No video stream found in {source}")

    if duration <= 0:
        raise FFmpegError(f"Could not determine positive duration for {source}")

    return VideoProbeInfo(
        duration_seconds=duration,
        has_video=has_video,
        has_audio=has_audio,
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
    )


def extract_frames_at_fps(
    media_path: PathLike,
    output_dir: PathLike,
    *,
    fps: float,
    ffmpeg_path: Optional[str] = None,
    image_ext: str = "jpg",
    timeout_seconds: int = 600,
) -> list[Path]:
    """Extract frames at a fixed FPS into ``output_dir``; return sorted frame paths."""
    source = Path(media_path)
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    pattern = dest_dir / f"frame_%05d.{image_ext}"

    exe = find_ffmpeg(ffmpeg_path)
    cmd = [
        exe,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(pattern),
    ]
    logger.info("Extracting frames fps=%s from %s -> %s", fps, source, dest_dir)
    _run_ffmpeg(cmd, timeout_seconds=timeout_seconds, source=source)

    frames = sorted(dest_dir.glob(f"frame_*.{image_ext}"))
    if not frames:
        raise FFmpegError(f"FFmpeg extracted no frames from {source}")
    return frames


def extract_frame_at_timestamp(
    media_path: PathLike,
    output_path: PathLike,
    *,
    timestamp_seconds: float,
    ffmpeg_path: Optional[str] = None,
    timeout_seconds: int = 120,
) -> Path:
    """Extract a single JPEG frame near ``timestamp_seconds`` via FFmpeg."""
    source = Path(media_path)
    if not source.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")
    if timestamp_seconds < 0:
        raise ValueError("timestamp_seconds must be >= 0")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    exe = find_ffmpeg(ffmpeg_path)
    # Place -ss before -i for fast seek; acceptable for keyframe approx sampling.
    cmd = [
        exe,
        "-y",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(dest),
    ]
    logger.info("Extracting frame at t=%.3fs -> %s", timestamp_seconds, dest)
    _run_ffmpeg(cmd, timeout_seconds=timeout_seconds, source=source)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise FFmpegError(
            f"FFmpeg produced no frame at t={timestamp_seconds:.3f}s for {source}",
        )
    return dest


def _run_ffmpeg(cmd: list[str], *, timeout_seconds: int, source: Path) -> None:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"FFmpeg timed out after {timeout_seconds}s while processing {source}",
        ) from exc
    except OSError as exc:
        raise FFmpegError(f"Failed to launch FFmpeg: {exc}") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        detail = stderr[-800:] if stderr else "no stderr"
        raise FFmpegError(
            f"FFmpeg failed (exit {completed.returncode}) for {source}: {detail}",
        )
