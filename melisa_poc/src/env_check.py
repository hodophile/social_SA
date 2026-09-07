"""Environment and dependency checks for the MyUni sentiment POC."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Optional

from src.config import (
    DEFAULT_ASR_LANGUAGE,
    DEFAULT_FUSION,
    DEFAULT_FUSION_YAML,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VIDEO_MAX_FRAMES,
    DEFAULT_VIDEO_MAX_OCR_FRAMES,
    DEFAULT_VIDEO_SAMPLE_FPS,
    DEFAULT_VIDEO_SAMPLING_STRATEGY,
    DEFAULT_VISUAL_MODEL,
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_MODEL,
)
from src.media.ffmpeg_utils import FFmpegNotFoundError, find_ffmpeg, find_ffprobe


def check_cuda_available() -> tuple[bool, str]:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, f"CUDA available ({name})"
        return False, "CUDA not available (CPU mode)"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as exc:  # noqa: BLE001
        return False, f"CUDA check failed: {exc}"


def check_ffmpeg() -> tuple[bool, str]:
    try:
        ffmpeg = find_ffmpeg()
        ffprobe = find_ffprobe()
        return True, f"FFmpeg OK ({ffmpeg}); ffprobe OK ({ffprobe})"
    except FFmpegNotFoundError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"FFmpeg check failed: {exc}"


def check_tesseract() -> tuple[bool, str]:
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError:
        return False, "pytesseract not installed (OCR disabled)"
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract available (version {version})"
    except TesseractNotFoundError:
        return False, (
            "Tesseract executable not found. Install Tesseract OCR and add it to PATH "
            "(Windows: https://github.com/UB-Mannheim/tesseract/wiki)."
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Tesseract check failed: {exc}"


def check_scenedetect() -> tuple[bool, str]:
    try:
        import scenedetect  # noqa: F401

        return True, "PySceneDetect available for scene_keyframe sampling"
    except ImportError:
        return False, "PySceneDetect not installed (scene_keyframe sampling unavailable)"


def configured_models() -> dict[str, str]:
    return {
        "text": DEFAULT_TEXT_MODEL,
        "visual": DEFAULT_VISUAL_MODEL,
        "asr": DEFAULT_WHISPER_MODEL,
        "asr_compute_type": DEFAULT_WHISPER_COMPUTE_TYPE,
        "asr_language": DEFAULT_ASR_LANGUAGE,
        "fusion": "poc-fusion",
    }


def video_sampling_defaults() -> dict[str, Any]:
    return {
        "default_strategy": DEFAULT_VIDEO_SAMPLING_STRATEGY,
        "fps": DEFAULT_VIDEO_SAMPLE_FPS,
        "max_frames": DEFAULT_VIDEO_MAX_FRAMES,
        "max_ocr_frames": DEFAULT_VIDEO_MAX_OCR_FRAMES,
    }


def check_transformers_version() -> tuple[bool, str]:
    try:
        import transformers

        version = transformers.__version__
        major = int(version.split(".", maxsplit=1)[0])
        if major >= 5:
            return (
                False,
                f"transformers {version} breaks SigLIP 2 loading; pin transformers>=4.45,<5 "
                "(see requirements.txt)",
            )
        return True, f"transformers {version}"
    except ImportError:
        return False, "transformers not installed"


def build_health_report(*, db_path: Optional[str] = None) -> dict[str, Any]:
    """Return a JSON-serializable environment report (no sensitive host details)."""
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    tess_ok, tess_msg = check_tesseract()
    cuda_ok, cuda_msg = check_cuda_available()
    scene_ok, scene_msg = check_scenedetect()
    transformers_ok, transformers_msg = check_transformers_version()

    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "dependencies": {
            "ffmpeg": {"available": ffmpeg_ok, "detail": ffmpeg_msg},
            "tesseract": {"available": tess_ok, "detail": tess_msg},
            "cuda": {"available": cuda_ok, "detail": cuda_msg},
            "scenedetect": {"available": scene_ok, "detail": scene_msg},
            "transformers": {"available": transformers_ok, "detail": transformers_msg},
        },
        "models": configured_models(),
        "video_sampling": video_sampling_defaults(),
        "fusion": {
            "source": DEFAULT_FUSION.source_path or str(DEFAULT_FUSION_YAML),
            "note": DEFAULT_FUSION.note,
        },
        "sqlite": {
            "default_db_example": "data/myuni_poc.db",
            "configured_path": db_path,
            "exists": Path(db_path).is_file() if db_path else None,
        },
        "note": (
            "POC evaluation defaults only — fusion weights and daily aggregates are "
            "NOT the final client business scoring methodology."
        ),
    }
