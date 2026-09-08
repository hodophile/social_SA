"""Presentation helpers for the Gradio Hugging Face client (no inference)."""

from __future__ import annotations

from html import escape
from typing import Any, Optional

from src.routing.input_router import CapabilityStatus, InputType
from src.schemas import SentimentEvidence
from src.ui.display import format_confidence_pct, format_probability_pct

BOTH_INPUTS_MESSAGE = "Please analyze one content item at a time."
NO_SPEECH_MESSAGE = "No meaningful speech was detected."
EMPTY_INPUT_MESSAGE = "Enter text or upload a supported image, audio, or video to analyze."

_LABEL_COLORS = {
    "positive": "#15803d",
    "neutral": "#475569",
    "negative": "#b91c1c",
}


def _color(label: Optional[str]) -> str:
    return _LABEL_COLORS.get((label or "").lower(), "#475569")


def _dist_html(probs: Optional[dict[str, Any]]) -> str:
    probs = probs or {}
    return f"""
    <div class="mu-dist">
      <div class="mu-dist-card">
        <div class="mu-dist-label">Positive</div>
        <div class="mu-dist-value">{escape(format_probability_pct(float(probs.get("positive", 0.0))))}</div>
      </div>
      <div class="mu-dist-card">
        <div class="mu-dist-label">Neutral</div>
        <div class="mu-dist-value">{escape(format_probability_pct(float(probs.get("neutral", 0.0))))}</div>
      </div>
      <div class="mu-dist-card">
        <div class="mu-dist-label">Negative</div>
        <div class="mu-dist-value">{escape(format_probability_pct(float(probs.get("negative", 0.0))))}</div>
      </div>
    </div>
    """


def _pill(label: str) -> str:
    color = _color(label)
    return (
        f'<span class="mu-pill" style="background:{color}18;color:{color};'
        f'border:1px solid {color}55;">● {escape(label.upper())}</span>'
    )


def _evidence_block(title: str, evidence: SentimentEvidence, model_name: str) -> str:
    return f"""
    <div class="mu-metric">
      <div class="mu-metric-label">{escape(title)}</div>
      {_pill(evidence.label)}
      <div class="mu-conf">Confidence <strong>{escape(format_confidence_pct(evidence.confidence))}</strong></div>
      <div class="mu-model">Model · {escape(model_name)}</div>
    </div>
    {_dist_html(evidence.probabilities)}
    """


def _shell(detected: str, body: str) -> str:
    return f"""
    <div class="mu-card">
      <div class="mu-kicker">Detected Input</div>
      <div class="mu-detected">{escape(detected)}</div>
      {body}
    </div>
    """


def _message_card(title: str, message: str, detected: Optional[str] = None) -> str:
    detected_html = ""
    if detected:
        detected_html = (
            f'<div class="mu-kicker">Detected Input</div>'
            f'<div class="mu-detected">{escape(detected)}</div>'
        )
    return f"""
    <div class="mu-card">
      {detected_html}
      <div class="mu-title">{escape(title)}</div>
      <p class="mu-copy">{escape(message)}</p>
    </div>
    """


def _ocr_unavailable(warnings: Optional[list[str]]) -> bool:
    joined = " ".join(warnings or []).lower()
    return "ocr unavailable" in joined or "tesseract" in joined


def render_idle() -> str:
    return """
    <div class="mu-card">
      <div class="mu-title">Analysis</div>
      <p class="mu-copy">Submit one text post or one media file. Content type is detected automatically.</p>
      <ul class="mu-list">
        <li>Text — Twitter-RoBERTa</li>
        <li>Image — SigLIP 2 visual sentiment</li>
        <li>Audio — Faster-Whisper transcript, then Twitter-RoBERTa</li>
        <li>Video — sampled frames (SigLIP 2) + optional speech, late fusion</li>
      </ul>
    </div>
    """


def render_validation(message: str) -> str:
    return _message_card("Unable to analyze", message)


def render_technical_details(routed: Any) -> str:
    """Client-safe technical notes. OCR gaps stay here, not in the main result."""
    detected = routed.detected_input
    kind = detected.value.upper() if detected else "UNKNOWN"
    lines = [
        f"**Detected input:** {kind}",
        f"**Display model:** {routed.model_display_name or '—'}",
        f"**Technical identifier:** `{routed.model_id or '—'}`",
    ]
    analysis = getattr(routed, "analysis", None)
    warnings: list[str] = []
    if analysis is not None:
        warnings = list(analysis.analysis.warnings or [])
        runtime = analysis.analysis.runtime
        if runtime is not None:
            models = runtime.models or {}
            if models.get("text"):
                lines.append(f"**Twitter-RoBERTa:** `{models['text']}`")
            if models.get("visual"):
                lines.append(f"**SigLIP 2:** `{models['visual']}`")
            if models.get("asr"):
                lines.append(
                    f"**Faster-Whisper:** `{models['asr']}` "
                    f"({models.get('asr_compute_type', 'int8')}, "
                    f"{models.get('asr_language', 'en')})"
                )
        if kind == "IMAGE":
            lines.append("**Visual checkpoint:** `google/siglip2-base-patch16-224`")
            if _ocr_unavailable(warnings):
                lines.append(
                    "**OCR:** unavailable in this environment. "
                    "Visual sentiment still ran; OCR text was not used."
                )
            elif analysis.analysis.ocr_text:
                lines.append("**OCR:** text extracted and scored when meaningful.")
            else:
                lines.append("**OCR:** no meaningful embedded text used.")
        if kind == "TEXT":
            lines.append(
                "**Text checkpoint:** `cardiffnlp/twitter-roberta-base-sentiment-latest`. "
                "Label is the highest class probability."
            )
        if kind == "AUDIO":
            lines.append(
                "**Audio path:** Faster-Whisper `base.en` (CPU int8) → Twitter-RoBERTa on the transcript."
            )
        if kind == "VIDEO":
            video = analysis.analysis.video
            if video is not None:
                lines.append(
                    f"**Sampling:** `{video.sampling_strategy}` · "
                    f"extracted {video.frames_extracted} · analyzed {video.frames_analyzed}"
                )
            lines.append(
                "**Video path:** FFmpeg frame sampling (CPU) · SigLIP 2 (ZeroGPU) · "
                "Faster-Whisper CPU int8 → Twitter-RoBERTa. Fusion is a POC baseline only."
            )
    if warnings:
        lines.append("**Pipeline notes:**")
        for warning in warnings[:8]:
            lines.append(f"- {warning}")
    return "\n\n".join(lines)


def render_routed_result(routed: Any) -> str:
    status = routed.status
    detected = routed.detected_input
    kind = detected.value.upper() if detected else "UNKNOWN"

    if status == CapabilityStatus.VALIDATION_ERROR:
        return render_validation(routed.message or "Unable to complete analysis.")

    if status == CapabilityStatus.INSUFFICIENT_EVIDENCE:
        if detected == InputType.AUDIO:
            return _shell(
                "AUDIO",
                f'<p class="mu-copy">{escape(NO_SPEECH_MESSAGE)}</p>'
                '<p class="mu-note">No transcript sentiment was invented.</p>',
            )
        return _message_card(
            "Insufficient evidence",
            routed.message or "Insufficient evidence for a sentiment result.",
            detected=kind,
        )

    if status != CapabilityStatus.OK or routed.analysis is None:
        return render_validation(routed.message or "Unable to complete analysis.")

    analysis = routed.analysis
    block = analysis.analysis
    modalities = block.modalities

    if kind == "TEXT" and modalities.text is not None:
        body = _evidence_block("Overall Sentiment", modalities.text, "Twitter-RoBERTa")
        return _shell("TEXT", body)

    if kind == "IMAGE" and modalities.visual is not None:
        body = _evidence_block("Visual Sentiment", modalities.visual, "SigLIP 2")
        return _shell("IMAGE", body)

    if kind == "AUDIO":
        if modalities.speech is None:
            return _shell(
                "AUDIO",
                f'<p class="mu-copy">{escape(NO_SPEECH_MESSAGE)}</p>',
            )
        transcript = block.transcript or ""
        transcript_html = (
            f'<div class="mu-sub">Transcript</div>'
            f'<blockquote class="mu-quote">{escape(transcript)}</blockquote>'
            if transcript
            else f'<p class="mu-copy">{escape(NO_SPEECH_MESSAGE)}</p>'
        )
        body = transcript_html + _evidence_block(
            "Transcript Sentiment",
            modalities.speech,
            "Faster-Whisper + Twitter-RoBERTa",
        )
        return _shell("AUDIO", body)

    if kind == "VIDEO":
        overall = block.overall
        used = list(block.fusion.contributing_modalities) if block.fusion else []
        visual_only = used == ["visual"]
        parts = [_evidence_block("Overall Sentiment", overall, routed.model_display_name or "POC fusion")]

        parts.append('<div class="mu-sub">Visual Evidence</div>')
        video = block.video
        strategy = video.sampling_strategy if video else "fixed_fps"
        analyzed = video.frames_analyzed if video else 0
        extracted = video.frames_extracted if video else 0
        parts.append(
            f'<p class="mu-copy">Strategy: <strong>{escape(str(strategy))}</strong> · '
            f"Frames sampled/analyzed: <strong>{extracted}/{analyzed}</strong></p>"
        )
        if modalities.visual is not None:
            parts.append(
                f'<div class="mu-inline">Visual sentiment {_pill(modalities.visual.label)}</div>'
            )
            parts.append(_dist_html(modalities.visual.probabilities))
        else:
            parts.append('<p class="mu-copy">No usable visual evidence.</p>')

        parts.append('<div class="mu-sub">Speech Evidence</div>')
        if block.transcript:
            parts.append(
                f'<blockquote class="mu-quote">{escape(block.transcript)}</blockquote>'
            )
        if modalities.speech is not None:
            parts.append(
                f'<div class="mu-inline">Speech sentiment {_pill(modalities.speech.label)}</div>'
            )
            parts.append(_dist_html(modalities.speech.probabilities))
        else:
            parts.append(f'<p class="mu-copy">{escape(NO_SPEECH_MESSAGE)}</p>')
            parts.append('<p class="mu-note">Speech was not treated as a neutral sentiment score.</p>')

        parts.append('<div class="mu-sub">Fusion</div>')
        used_label = ", ".join(used) if used else "none"
        parts.append(
            f'<p class="mu-copy">Available modalities used: <strong>{escape(used_label)}</strong></p>'
        )
        if visual_only:
            parts.append(
                '<p class="mu-note">Overall sentiment used visual evidence only. '
                "It is not a speech/neutral default.</p>"
            )
        if block.fusion and block.fusion.explanation:
            parts.append(f'<p class="mu-copy">{escape(block.fusion.explanation)}</p>')
        return _shell("VIDEO", "".join(parts))

    evidence = modalities.text or modalities.visual or modalities.speech or block.overall
    body = _evidence_block("Overall Sentiment", evidence, routed.model_display_name or "—")
    return _shell(kind, body)
