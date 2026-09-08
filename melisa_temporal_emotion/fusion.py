"""
Fusion helpers for the temporal-emotion pipeline.

Melisa's own fuse_modalities only produces a 3-class output
(positive/neutral/negative), so we add fuse_emotion_distributions
which merges per-modality 8-emotion distributions using the same
confidence-weighted pattern:

    effective_weight_i = configured_weight_i * confidence_i
    fused = sum(dist_i * effective_weight_i) / sum(effective_weight_i)
"""

import math
from typing import Dict, List, Optional, Tuple

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

VALENCE_MAP = {
    "joy": 1.0,
    "trust": 0.5,
    "anticipation": 0.3,
    "surprise": 0.0,
    "fear": -0.5,
    "anger": -0.8,
    "sadness": -1.0,
    "disgust": -0.8,
}

DEFAULT_MODALITY_WEIGHTS = {
    "text": 1.0,
    "visual": 1.0,
    "speech": 1.0,
}


def temporal_weighted_average(
    emotion_seqs: List[Dict[str, float]],
    half_life_seconds: float = 2.0,
    fps: float = 1.0,
) -> Dict[str, float]:
    """Exponential-decay average of per-frame emotion distributions.

    More recent frames weigh more, which encodes temporal order
    (something the original Melisa frame aggregation does not do).
    """
    if not emotion_seqs:
        return {e: 0.0 for e in EMOTION_LABELS}

    weights = []
    for i in range(len(emotion_seqs)):
        t = i / fps
        weights.append(math.exp(-math.log(2) * t / half_life_seconds))

    w_sum = sum(weights)
    norm = [w / w_sum for w in weights]

    agg = {e: 0.0 for e in EMOTION_LABELS}
    for emo_dict, w in zip(emotion_seqs, norm):
        for e, v in emo_dict.items():
            if e in agg:
                agg[e] += w * v

    return agg


def fuse_emotion_distributions(
    modalities: Dict[str, Optional[Tuple[Dict[str, float], float]]],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float]:
    """Confidence-weighted fusion of per-modality 8-emotion distributions.

    modalities: {"text": (dist, confidence) | None, ...}
    Returns (fused_distribution, fused_confidence).
    """
    weights = weights or DEFAULT_MODALITY_WEIGHTS

    num = {e: 0.0 for e in EMOTION_LABELS}
    w_sum = 0.0
    c_acc = 0.0

    for name, item in modalities.items():
        if not item:
            continue
        dist, confidence = item
        eff = float(weights.get(name, 1.0)) * max(0.0, float(confidence))
        if eff <= 0:
            continue
        w_sum += eff
        c_acc += eff * float(confidence)
        for e in EMOTION_LABELS:
            num[e] += eff * float(dist.get(e, 0.0))

    if w_sum <= 0:
        return ({e: 1.0 / len(EMOTION_LABELS) for e in EMOTION_LABELS}, 0.0)

    fused = {e: num[e] / w_sum for e in EMOTION_LABELS}
    total = sum(fused.values())
    if total > 0:
        fused = {e: v / total for e, v in fused.items()}

    return fused, min(1.0, c_acc / w_sum)


def valence_of(distribution: Dict[str, float]) -> float:
    """Map an 8-emotion distribution onto a -1..+1 valence score."""
    return round(
        sum(distribution.get(e, 0.0) * VALENCE_MAP[e] for e in EMOTION_LABELS),
        3,
    )
