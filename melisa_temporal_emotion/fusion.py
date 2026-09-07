"""
Temporal‑weighted average helper used by video_analyzer.py.
The rest of the fusion logic (confidence‑weighted late fusion) is exactly
the same as in Melisa POC, so we reuse melisa_poc.src.fusion.fuse_modalities.
"""
import math
from typing import List, Dict

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

def temporal_weighted_average(
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
