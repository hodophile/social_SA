"""
melisa_bridge.py — bridge to the MyUni Melisa sentiment POC
(space: ajayck007/myuni-sentiment-poc-melisa-v1), vendored
under melisa_poc/.

Runs their pipeline alongside ours for comparative deployment
and maps their final sentiment labels onto our emotion labels
(Plutchik's 8 basic emotions, see app.py / METRICS.md).

Their fusion (melisa_poc/src/fusion.py, config/fusion.yaml):

    effective_weight_i = configured_weight_i * confidence_i
    fused_score = sum(score_i * effective_weight_i)
                  / sum(effective_weight_i)

    label = positive if score >  0.15
            negative if score < -0.15
            neutral  otherwise

with a confidence penalty when high-confidence modalities
disagree (modality conflict detection).
"""

from __future__ import annotations

import mimetypes
import sys
from pathlib import Path
from typing import Any, Optional

MELISA_ROOT = Path(__file__).resolve().parent / "melisa_poc"

if str(MELISA_ROOT) not in sys.path:
    sys.path.insert(0, str(MELISA_ROOT))

# ---------------------------------------------------------------------------
# Label matching: Melisa sentiment labels -> our emotion labels.
# ---------------------------------------------------------------------------
# Melisa fuses modality evidence into positive / neutral / negative.
# Our emotion schema uses Plutchik's 8 basic emotions, so the final
# labels are matched onto the closest emotion representative:
#
#   positive -> joy     (pleasant / uplifting content)
#   neutral  -> trust   (even, informational, no polarity)
#   negative -> sadness (unpleasant / distressing content)
#
# The mapping is intentionally explicit and easy to change.

MELISA_TO_EMOTION = {
    "positive": "joy",
    "neutral": "trust",
    "negative": "sadness",
}

OUR_EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
]

_pipeline = None


def get_melisa_pipeline():
    """Lazy-load the vendored Melisa pipeline (models load on demand)."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import MyUniSentimentPipeline  # vendored src/

        _pipeline = MyUniSentimentPipeline()
    return _pipeline


def analyze_with_melisa(
    text: Optional[str],
    media_path: Optional[str],
    filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the Melisa pipeline and return a comparison-ready dict:

    - raw Melisa sentiment (label / score / confidence / fusion)
    - final labels matched to our emotion schema
    - valence on the same -1..+1 scale as ours (their fused score)
    """

    mime_type = None
    if media_path:
        name = filename or Path(media_path).name
        mime_type, _ = mimetypes.guess_type(name)

    try:
        routed = get_melisa_pipeline().analyze(
            text=text,
            media_path=media_path,
            mime_type=mime_type,
            filename=filename,
            user_id="COMPARE-USER",
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    status = getattr(routed.status, "value", str(routed.status))

    if status != "ok":
        return {
            "status": status,
            "message": routed.message,
        }

    # ActivityAnalysisResult -> AnalysisBlock -> overall SentimentEvidence
    analysis = routed.analysis.analysis
    overall = analysis.overall
    fusion = analysis.fusion

    label = overall.label  # positive | neutral | negative
    matched_emotion = MELISA_TO_EMOTION.get(label, "trust")

    # Project their label probabilities onto our emotion schema.
    probs = overall.probabilities or {}
    emotion_distribution = {e: 0.0 for e in OUR_EMOTION_LABELS}
    for melisa_label, p in probs.items():
        emotion = MELISA_TO_EMOTION.get(melisa_label)
        if emotion:
            emotion_distribution[emotion] = round(
                emotion_distribution[emotion] + float(p), 3
            )

    return {
        "status": "ok",
        "detected_input": getattr(
            routed.detected_input, "value", str(routed.detected_input)
        ),
        # ---- raw Melisa output ----
        "initial_label": label,
        "initial_score": overall.score,
        "initial_confidence": overall.confidence,
        "initial_probabilities": probs,
        # ---- matched to our emotion labels ----
        "matched_emotion": matched_emotion,
        "label_mapping": MELISA_TO_EMOTION,
        "emotion_distribution": emotion_distribution,
        # their fused score is already in [-1, +1] -> directly
        # comparable to our valence
        "valence": overall.score,
        # ---- fusion transparency ----
        "fusion": {
            "contributing_modalities": (
                fusion.contributing_modalities if fusion else []
            ),
            "modality_conflict": (
                fusion.modality_conflict if fusion else False
            ),
            "disagreement_score": (
                fusion.disagreement_score if fusion else 0.0
            ),
            "explanation": fusion.explanation if fusion else "",
        },
        "warnings": analysis.warnings,
    }
