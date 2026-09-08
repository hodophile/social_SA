"""POC daily user-level sentiment aggregation (not client business scoring)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from src.config import DEFAULT_FUSION, FusionConfig
from src.fusion import score_to_label
from src.schemas import DailyUserScore
from src.storage.repository import SentimentRepository


POC_DAILY_NOTE = (
    "POC daily aggregate — mean of stored activity sentiment scores; "
    "NOT the future client business score"
)


def compute_daily_user_score(
    *,
    user_id: str,
    score_date: str,
    rows: Iterable[dict],
    activity_count: Optional[int] = None,
    config: FusionConfig = DEFAULT_FUSION,
) -> DailyUserScore:
    """Compute POC daily aggregate fields from processed analysis rows."""
    processed = [r for r in rows if r.get("status") == "processed" and r.get("overall_score") is not None]
    pos = neu = neg = 0
    scores: list[float] = []
    for row in processed:
        label = (row.get("overall_label") or "").lower()
        score = float(row["overall_score"])
        scores.append(score)
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1

    mean_score = (sum(scores) / len(scores)) if scores else None
    daily_label = score_to_label(mean_score, config) if mean_score is not None else None

    return DailyUserScore(
        user_id=user_id,
        score_date=score_date,
        activity_count=activity_count if activity_count is not None else len(list(rows)),
        valid_analysis_count=len(processed),
        mean_sentiment_score=mean_score,
        positive_count=pos,
        neutral_count=neu,
        negative_count=neg,
        daily_sentiment_label=daily_label,
        updated_at=datetime.now(timezone.utc),
        note=POC_DAILY_NOTE,
    )


def refresh_daily_user_score(
    repo: SentimentRepository,
    *,
    user_id: str,
    score_date: str,
    config: FusionConfig = DEFAULT_FUSION,
) -> DailyUserScore:
    """Recompute and persist POC daily aggregate for one user/date."""
    processed_rows = repo.list_processed_for_user_date(user_id, score_date)
    activity_count = repo.count_activities_for_user_date(user_id, score_date)
    score = compute_daily_user_score(
        user_id=user_id,
        score_date=score_date,
        rows=processed_rows,
        activity_count=activity_count,
        config=config,
    )
    repo.upsert_daily_user_score(score)
    return score
