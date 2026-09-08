"""
qwen_bridge.py — Bridge from Qwen2.5-VLM-3B pipeline to comparison JSON format.

Wraps social_media_analyzer.SocialMediaAnalyzer and converts its output
to the same shape used by the temporal-emotion pipeline so both can be
compared side-by-side in the Gradio UI.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from social_media_analyzer import (
    SocialMediaAnalyzer,
    SocialMediaPost,
    QwenVLM,
    HarmClassifier,
)

# Map free-form Qwen tones to our 8 Plutchik emotions
TONE_TO_EMOTION = {
    "joyful": "joy",
    "happy": "joy",
    "excited": "joy",
    "angry": "anger",
    "furious": "anger",
    "sad": "sadness",
    "depressed": "sadness",
    "fearful": "fear",
    "scared": "fear",
    "anxious": "fear",
    "surprised": "surprise",
    "shocked": "surprise",
    "disgusted": "disgust",
    "neutral": "trust",
    "informational": "trust",
    "calm": "trust",
    "sarcastic": "anticipation",
}

_pipeline = None


def get_qwen_pipeline(mock: bool = True):
    """Lazy-load the Qwen pipeline."""
    global _pipeline
    if _pipeline is None:
        vlm = QwenVLM(mock=mock)
        classifier = HarmClassifier()
        _pipeline = SocialMediaAnalyzer(vlm=vlm, classifier=classifier)
    return _pipeline


def _tone_to_emotion(tone: str) -> str:
    """Map a free-form tone string to the closest Plutchik emotion."""
    tone_lower = tone.lower().strip()
    # Direct mapping
    if tone_lower in TONE_TO_EMOTION:
        return TONE_TO_EMOTION[tone_lower]
    # Substring matching
    for key, emotion in TONE_TO_EMOTION.items():
        if key in tone_lower or tone_lower in key:
            return emotion
    return "trust"  # default fallback


def _build_emotion_analysis(understanding: dict) -> dict:
    """Build emotion_analysis block from Qwen understanding output."""
    tone = understanding.get("tone", "neutral")
    primary = _tone_to_emotion(tone)

    # Build a simple distribution: primary gets 0.6, rest uniform
    scores = {e: 0.0 for e in [
        "joy", "sadness", "anger", "fear",
        "surprise", "disgust", "trust", "anticipation"
    ]}
    scores[primary] = 0.6
    remaining = 0.4 / 7
    for e in scores:
        if e != primary:
            scores[e] = round(remaining, 3)

    # Normalize
    total = sum(scores.values())
    scores = {k: round(v / total, 3) for k, v in scores.items()}

    # Determine valence from primary emotion
    valence_map = {
        "joy": 0.8, "trust": 0.4, "anticipation": 0.3,
        "surprise": 0.0, "fear": -0.5, "anger": -0.7,
        "sadness": -0.9, "disgust": -0.7,
    }
    valence = valence_map.get(primary, 0.0)

    return {
        "emotion_model": "qwen2.5-vlm-3b (mock) + tone heuristic",
        "primary_emotion": primary,
        "secondary_emotion": None,
        "emotion_scores": scores,
        "emotion_distribution": scores,
        "valence": valence,
        "confidence": 0.7,
        "primary_emotion_score": round(scores[primary] * 0.7, 3),
        "rationale": (
            f"Qwen VLM detected tone '{tone}' → mapped to '{primary}'. "
            "Heuristic distribution used (no fine-grained emotion scores from VLM)."
        ),
    }


def analyze_with_qwen(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    sample_fps: float = 1.0,
    mock: bool = True,
) -> dict[str, Any]:
    """
    Run the Qwen2.5-VLM-3B pipeline and return comparison-ready dict.
    """
    t0 = time.perf_counter()

    post = SocialMediaPost(
        post_id="qwen-post",
        text=text,
        image_path=Path(image_path) if image_path else None,
        video_path=Path(video_path) if video_path else None,
    )

    try:
        result = get_qwen_pipeline(mock=mock).analyze(
            post=post,
            sample_fps=sample_fps,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    understanding = result.get("understanding", {})
    risk = result.get("risk", {})
    action = result.get("action", "ALLOW")

    emotion_analysis = _build_emotion_analysis(understanding)

    return {
        "execution_time_seconds": round(time.perf_counter() - t0, 3),
        "post_id": result.get("post_id", "qwen-post"),
        "understanding": understanding,
        "risk": risk,
        "action": action,
        "emotion_analysis": emotion_analysis,
    }
