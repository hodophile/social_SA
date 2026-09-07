"""Presentation helpers for the Streamlit demo (no inference)."""

from __future__ import annotations

from typing import Any, Optional

from src.schemas import ActivityAnalysisResult, FusionDiagnostics, SentimentEvidence

LABEL_COLORS = {
    "positive": "#1B7F4E",
    "neutral": "#5C6570",
    "negative": "#B42318",
}


def label_color(label: str) -> str:
    return LABEL_COLORS.get(label, "#5C6570")


def format_score(score: float) -> str:
    return f"{score:+.3f}"


def format_confidence(confidence: float) -> str:
    return f"{confidence:.3f}"


def format_confidence_pct(confidence: float) -> str:
    """Client-facing confidence as XX.X%."""
    return f"{confidence * 100:.1f}%"


def format_probability_pct(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def evidence_to_dict(ev: Optional[SentimentEvidence]) -> Optional[dict[str, Any]]:
    if ev is None:
        return None
    return {
        "label": ev.label,
        "score": ev.score,
        "confidence": ev.confidence,
        "probabilities": ev.probabilities,
        "model": ev.model,
    }


def humanize_ffmpeg_error(exc: BaseException) -> str:
    msg = str(exc)
    if "FFmpeg" in msg or "ffprobe" in msg.lower():
        return (
            f"{msg}\n\n"
            "Action: install FFmpeg and ensure `ffmpeg` / `ffprobe` are on PATH "
            "(Windows: `winget install Gyan.FFmpeg`, then reopen the terminal)."
        )
    return msg


def humanize_dependency_hint(warnings: list[str]) -> Optional[str]:
    """Return an actionable hint when OCR/FFmpeg issues appear in pipeline warnings."""
    joined = " ".join(warnings).lower()
    hints: list[str] = []
    if "tesseract" in joined or "ocr unavailable" in joined:
        hints.append(
            "OCR needs Tesseract on PATH "
            "(Windows: https://github.com/UB-Mannheim/tesseract/wiki).",
        )
    if "ffmpeg" in joined:
        hints.append(
            "Video/speech needs FFmpeg on PATH (`winget install Gyan.FFmpeg`).",
        )
    if not hints:
        return None
    return " ".join(hints)


def fusion_summary(fusion: Optional[FusionDiagnostics]) -> Optional[dict[str, Any]]:
    if fusion is None:
        return None
    return {
        "modality_conflict": fusion.modality_conflict,
        "disagreement_score": fusion.disagreement_score,
        "contributing_modalities": list(fusion.contributing_modalities),
        "explanation": fusion.explanation,
        "note": fusion.note,
    }


def overall_headline(result: ActivityAnalysisResult) -> dict[str, Any]:
    overall = result.analysis.overall
    return {
        "label": overall.label,
        "score": overall.score,
        "confidence": overall.confidence,
        "model": overall.model,
    }
