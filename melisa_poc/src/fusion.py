"""Explainable multimodal late fusion for the MyUni sentiment POC.

POC-only: weights and thresholds come from ``config/fusion.yaml`` and are
evaluation defaults — not client business scoring rules and not scientifically
validated.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.config import DEFAULT_FUSION, FusionConfig
from src.schemas import FusionDiagnostics, SentimentEvidence, SentimentLabel


def score_to_label(score: float, config: FusionConfig = DEFAULT_FUSION) -> SentimentLabel:
    """Map a fused score to a label using configurable POC thresholds."""
    if score > config.thresholds.positive_above:
        return "positive"
    if score < config.thresholds.negative_below:
        return "negative"
    return "neutral"


def _build_explanation(
    *,
    label: SentimentLabel,
    score: float,
    used: list[str],
    modality_conflict: bool,
    disagreement_score: float,
    config: FusionConfig,
) -> str:
    parts = [
        f"Late fusion over {', '.join(used) if used else 'no modalities'}",
        f"score={score:.3f}",
        f"label={label}",
        (
            f"thresholds=({config.thresholds.negative_below:.2f},"
            f"{config.thresholds.positive_above:.2f})"
        ),
    ]
    if modality_conflict:
        parts.append(f"modality_conflict=true disagreement={disagreement_score:.3f}")
    else:
        parts.append("modality_conflict=false")
    return "; ".join(parts)


def detect_modality_conflict(
    modalities: dict[str, SentimentEvidence],
    config: FusionConfig = DEFAULT_FUSION,
) -> tuple[bool, float, list[str]]:
    """Detect strong opposing evidence among high-confidence modalities.

    Returns ``(conflict, disagreement_score, strong_modality_names)``.
    ``disagreement_score`` is max(score)-min(score) among strong modalities (0 if <2).
    """
    strong: list[tuple[str, float]] = []
    for name, evidence in modalities.items():
        if evidence.confidence < config.conflict.min_confidence:
            continue
        if abs(evidence.score) < config.conflict.min_polarity:
            continue
        strong.append((name, evidence.score))

    if len(strong) < 2:
        return False, 0.0, [n for n, _ in strong]

    scores = [s for _, s in strong]
    disagreement = float(max(scores) - min(scores))
    has_pos = any(s > 0 for s in scores)
    has_neg = any(s < 0 for s in scores)
    conflict = (
        disagreement >= config.conflict.disagreement_threshold
        and has_pos
        and has_neg
    )
    return conflict, disagreement, [n for n, _ in strong]


def fuse_modality_scores(
    modalities: dict[str, Optional[SentimentEvidence]],
    config: FusionConfig = DEFAULT_FUSION,
) -> SentimentEvidence:
    """Transparent late fusion with conflict-aware confidence and diagnostics.

    Base formula::

        effective_weight_i = configured_weight_i * confidence_i
        fused_score = sum(score_i * effective_weight_i) / sum(effective_weight_i)

    Missing / zero-weight / zero-confidence modalities are skipped.
    """
    result = fuse_modalities(modalities, config=config)
    return result.overall


def fuse_modalities(
    modalities: dict[str, Optional[SentimentEvidence]],
    config: FusionConfig = DEFAULT_FUSION,
) -> "FusionOutcome":
    """Full fusion outcome including structured diagnostics."""
    configured_weights = {k: float(v) for k, v in config.modality_weights.items()}
    effective_weights: dict[str, float] = {}
    present: dict[str, SentimentEvidence] = {}

    weighted_score_num = 0.0
    weight_sum = 0.0
    conf_acc = 0.0

    for name, evidence in modalities.items():
        if evidence is None:
            continue
        configured = float(configured_weights.get(name, 0.0))
        if configured <= 0:
            continue
        eff = configured * float(evidence.confidence)
        if eff <= 0:
            continue
        present[name] = evidence
        effective_weights[name] = eff
        weighted_score_num += eff * float(evidence.score)
        weight_sum += eff
        conf_acc += eff * float(evidence.confidence)

    used = list(present.keys())

    if weight_sum <= 0:
        diagnostics = FusionDiagnostics(
            contributing_modalities=[],
            configured_weights=configured_weights,
            effective_weights={},
            modality_conflict=False,
            disagreement_score=0.0,
            thresholds={
                "positive_above": config.thresholds.positive_above,
                "negative_below": config.thresholds.negative_below,
            },
            explanation="No usable modality evidence; defaulting to neutral with confidence 0",
            note=config.note,
            source_path=config.source_path,
        )
        overall = SentimentEvidence(
            label="neutral",
            score=0.0,
            confidence=0.0,
            probabilities={"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            model="poc-fusion",
            details=diagnostics.model_dump(),
        )
        return FusionOutcome(overall=overall, diagnostics=diagnostics)

    fused_score = max(-1.0, min(1.0, weighted_score_num / weight_sum))
    fused_confidence = max(0.0, min(1.0, conf_acc / weight_sum))

    conflict, disagreement, _strong = detect_modality_conflict(present, config=config)
    if conflict:
        penalty = max(0.0, min(1.0, config.conflict.confidence_penalty))
        fused_confidence = max(0.0, min(1.0, fused_confidence * (1.0 - penalty)))

    label = score_to_label(fused_score, config=config)
    explanation = _build_explanation(
        label=label,
        score=fused_score,
        used=used,
        modality_conflict=conflict,
        disagreement_score=disagreement,
        config=config,
    )

    diagnostics = FusionDiagnostics(
        contributing_modalities=used,
        configured_weights=configured_weights,
        effective_weights={k: float(v) for k, v in effective_weights.items()},
        modality_conflict=conflict,
        disagreement_score=float(disagreement),
        thresholds={
            "positive_above": config.thresholds.positive_above,
            "negative_below": config.thresholds.negative_below,
        },
        explanation=explanation,
        note=config.note,
        source_path=config.source_path,
    )

    # Soft label distribution for transparency (not a calibrated probability).
    if label == "positive":
        probs = {"negative": 0.0, "neutral": max(0.0, 1.0 - fused_confidence), "positive": fused_confidence}
    elif label == "negative":
        probs = {"negative": fused_confidence, "neutral": max(0.0, 1.0 - fused_confidence), "positive": 0.0}
    else:
        probs = {"negative": 0.0, "neutral": 1.0, "positive": 0.0}

    overall = SentimentEvidence(
        label=label,
        score=float(fused_score),
        confidence=float(fused_confidence),
        probabilities=probs,
        model="poc-fusion",
        details=diagnostics.model_dump(),
    )
    return FusionOutcome(overall=overall, diagnostics=diagnostics)


class FusionOutcome:
    """Bundle overall sentiment evidence with structured fusion diagnostics."""

    __slots__ = ("overall", "diagnostics")

    def __init__(self, overall: SentimentEvidence, diagnostics: FusionDiagnostics) -> None:
        self.overall = overall
        self.diagnostics = diagnostics


def aggregate_frame_visual_scores(
    frame_evidence: Sequence[SentimentEvidence],
    *,
    config: FusionConfig = DEFAULT_FUSION,
    neutral_band: Optional[float] = None,
) -> Optional[SentimentEvidence]:
    """Confidence-weighted average of per-frame visual scores (no temporal model)."""
    if not frame_evidence:
        return None

    # Frame aggregation uses the same transparent weight=confidence pattern.
    band = config.neutral_band if neutral_band is None else neutral_band
    frame_cfg = FusionConfig(
        modality_weights={f"frame_{i}": 1.0 for i in range(len(frame_evidence))},
        thresholds=config.thresholds,
        conflict=config.conflict,
        note="POC frame visual aggregate; not a temporal model",
        source_path=config.source_path,
    )
    modalities = {f"frame_{i}": ev for i, ev in enumerate(frame_evidence)}
    outcome = fuse_modalities(modalities, config=frame_cfg)
    # Preserve a stable model id from frames when possible.
    models = [ev.model for ev in frame_evidence if ev.model]
    model_id = models[0] if models else "frame-aggregate"
    details = dict(outcome.overall.details or {})
    details.update(
        {
            "method": "confidence-weighted average over sampled frames",
            "frames_aggregated": len(frame_evidence),
            "neutral_band": band,
            "note": "POC video visual summary; not a trained temporal model",
        },
    )
    return outcome.overall.model_copy(update={"model": model_id, "details": details})
