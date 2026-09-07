"""SQLite repository for activities, analysis results, batches, and daily scores."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from src.schemas import ActivityAnalysisResult, ActivityInput, DailyUserScore
from src.storage.schema import SCHEMA_SQL

PathLike = Union[str, Path]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def score_date_from_timestamp(ts: datetime | str | None) -> str:
    """UTC calendar date used for POC daily aggregation (processing date)."""
    if ts is None:
        return datetime.now(timezone.utc).date().isoformat()
    if isinstance(ts, datetime):
        dt = ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().isoformat()
    # ISO string
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).date().isoformat()
    except ValueError:
        return str(ts)[:10]


class SentimentRepository:
    """Minimal SQLite access layer for the MyUni sentiment POC."""

    def __init__(self, db_path: PathLike) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent and str(self.db_path.parent) not in {"", "."}:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------ batch
    def start_batch_run(self, *, source_path: str) -> str:
        batch_id = f"BATCH-{uuid.uuid4().hex[:12].upper()}"
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO batch_runs (
                    batch_id, source_path, started_at, status,
                    total_records, succeeded, failed, invalid, unsupported, skipped
                ) VALUES (?, ?, ?, 'running', 0, 0, 0, 0, 0, 0)
                """,
                (batch_id, source_path, _utc_now_iso()),
            )
        return batch_id

    def complete_batch_run(
        self,
        batch_id: str,
        *,
        status: str,
        total_records: int,
        succeeded: int,
        failed: int,
        invalid: int = 0,
        unsupported: int = 0,
        skipped: int = 0,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE batch_runs
                SET completed_at = ?, status = ?, total_records = ?,
                    succeeded = ?, failed = ?, invalid = ?, unsupported = ?, skipped = ?
                WHERE batch_id = ?
                """,
                (
                    _utc_now_iso(),
                    status,
                    total_records,
                    succeeded,
                    failed,
                    invalid,
                    unsupported,
                    skipped,
                    batch_id,
                ),
            )

    def get_batch_run(self, batch_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM batch_runs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        return dict(row) if row else None

    # -------------------------------------------------------------- activities
    def activity_exists(self, activity_id: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM activities WHERE activity_id = ?",
                (activity_id,),
            ).fetchone()
        return row is not None

    def get_analysis_status(self, activity_id: str) -> Optional[str]:
        """Return stored analysis status ('processed', 'failed', ...) or None."""
        row = self.get_analysis(activity_id)
        if row is None:
            return None
        status = row.get("status")
        return str(status) if status is not None else None

    def has_processed_analysis(self, activity_id: str) -> bool:
        return self.get_analysis_status(activity_id) == "processed"

    def insert_activity(
        self,
        activity: ActivityInput,
        *,
        batch_id: Optional[str] = None,
    ) -> bool:
        """Insert activity. Returns False if activity_id already exists (no silent overwrite)."""
        created = activity.created_at.isoformat() if activity.created_at else None
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO activities (
                        activity_id, user_id, activity_type, text, media_path,
                        created_at, metadata_json, batch_id, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        activity.activity_id,
                        activity.user_id,
                        activity.activity_type,
                        activity.text,
                        activity.media_path,
                        created,
                        _json_dumps(activity.metadata),
                        batch_id,
                        _utc_now_iso(),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    # ------------------------------------------------------- analysis results
    def upsert_analysis_result(
        self,
        *,
        activity_id: str,
        user_id: str,
        status: str,
        batch_id: Optional[str] = None,
        result: Optional[ActivityAnalysisResult] = None,
        error: Optional[str] = None,
        processed_at: Optional[datetime] = None,
    ) -> str:
        """Store analysis row for an activity. Returns score_date used."""
        processed = processed_at or datetime.now(timezone.utc)
        processed_iso = processed.isoformat()
        score_date = score_date_from_timestamp(processed)

        overall_label = overall_score = overall_confidence = None
        modalities_json = fusion_json = models_json = warnings_json = result_json = None

        if result is not None:
            overall = result.analysis.overall
            overall_label = overall.label
            overall_score = overall.score
            overall_confidence = overall.confidence
            modalities_json = _json_dumps(
                result.analysis.modalities.model_dump(mode="json"),
            )
            fusion_json = _json_dumps(
                result.analysis.fusion.model_dump(mode="json")
                if result.analysis.fusion
                else overall.details,
            )
            models = {
                "overall": overall.model,
            }
            mods = result.analysis.modalities
            for name in ("text", "visual", "ocr", "speech"):
                ev = getattr(mods, name, None)
                if ev is not None and ev.model:
                    models[name] = ev.model
            models_json = _json_dumps(models)
            warnings_json = _json_dumps(result.analysis.warnings)
            result_json = _json_dumps(result.model_dump_json_compatible())
            score_date = score_date_from_timestamp(result.processed_at)

        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_results (
                    activity_id, user_id, overall_label, overall_score, overall_confidence,
                    modalities_json, fusion_json, models_json, warnings_json,
                    status, error, processed_at, score_date, batch_id, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(activity_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    overall_label=excluded.overall_label,
                    overall_score=excluded.overall_score,
                    overall_confidence=excluded.overall_confidence,
                    modalities_json=excluded.modalities_json,
                    fusion_json=excluded.fusion_json,
                    models_json=excluded.models_json,
                    warnings_json=excluded.warnings_json,
                    status=excluded.status,
                    error=excluded.error,
                    processed_at=excluded.processed_at,
                    score_date=excluded.score_date,
                    batch_id=excluded.batch_id,
                    result_json=excluded.result_json
                """,
                (
                    activity_id,
                    user_id,
                    overall_label,
                    overall_score,
                    overall_confidence,
                    modalities_json,
                    fusion_json,
                    models_json,
                    warnings_json,
                    status,
                    error,
                    processed_iso,
                    score_date,
                    batch_id,
                    result_json,
                ),
            )
        return score_date

    def get_analysis(self, activity_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_results WHERE activity_id = ?",
                (activity_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_processed_for_user_date(
        self,
        user_id: str,
        score_date: str,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM analysis_results
                WHERE user_id = ? AND score_date = ? AND status = 'processed'
                ORDER BY processed_at
                """,
                (user_id, score_date),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_activities_for_user_date(self, user_id: str, score_date: str) -> int:
        """Count activities for user whose analysis score_date matches (any status)."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM analysis_results
                WHERE user_id = ? AND score_date = ?
                """,
                (user_id, score_date),
            ).fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------ daily user scores
    def upsert_daily_user_score(self, score: DailyUserScore) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_user_scores (
                    user_id, score_date, activity_count, valid_analysis_count,
                    mean_sentiment_score, positive_count, neutral_count, negative_count,
                    daily_sentiment_label, updated_at, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, score_date) DO UPDATE SET
                    activity_count=excluded.activity_count,
                    valid_analysis_count=excluded.valid_analysis_count,
                    mean_sentiment_score=excluded.mean_sentiment_score,
                    positive_count=excluded.positive_count,
                    neutral_count=excluded.neutral_count,
                    negative_count=excluded.negative_count,
                    daily_sentiment_label=excluded.daily_sentiment_label,
                    updated_at=excluded.updated_at,
                    note=excluded.note
                """,
                (
                    score.user_id,
                    score.score_date,
                    score.activity_count,
                    score.valid_analysis_count,
                    score.mean_sentiment_score,
                    score.positive_count,
                    score.neutral_count,
                    score.negative_count,
                    score.daily_sentiment_label,
                    score.updated_at.isoformat(),
                    score.note,
                ),
            )

    def get_daily_scores(
        self,
        *,
        score_date: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[DailyUserScore]:
        clauses: list[str] = []
        params: list[str] = []
        if score_date:
            clauses.append("score_date = ?")
            params.append(score_date)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM daily_user_scores {where} ORDER BY score_date, user_id"
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[DailyUserScore] = []
        for row in rows:
            out.append(
                DailyUserScore(
                    user_id=row["user_id"],
                    score_date=row["score_date"],
                    activity_count=row["activity_count"],
                    valid_analysis_count=row["valid_analysis_count"],
                    mean_sentiment_score=row["mean_sentiment_score"],
                    positive_count=row["positive_count"],
                    neutral_count=row["neutral_count"],
                    negative_count=row["negative_count"],
                    daily_sentiment_label=row["daily_sentiment_label"],
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    note=row["note"],
                ),
            )
        return out
