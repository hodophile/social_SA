"""Batch processing with SQLite persistence and daily aggregation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from src.batch import _format_validation_error, parse_activity_record
from src.config import DEFAULT_FUSION, FusionConfig
from src.pipeline import MyUniSentimentPipeline
from src.schemas import BatchProcessingResult, BatchRecordOutcome, BatchSummary
from src.storage.aggregation import refresh_daily_user_score
from src.storage.repository import SentimentRepository

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class PersistentBatchRunner:
    """JSONL batch → analysis → SQLite activity/result storage → daily POC aggregates."""

    def __init__(
        self,
        db_path: PathLike,
        *,
        pipeline: Optional[MyUniSentimentPipeline] = None,
        fusion_config: FusionConfig = DEFAULT_FUSION,
    ) -> None:
        self.repo = SentimentRepository(db_path)
        self.pipeline = pipeline or MyUniSentimentPipeline()
        self.fusion_config = fusion_config

    def process_file(self, path: PathLike) -> BatchProcessingResult:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Batch file not found: {file_path}")

        batch_id = self.repo.start_batch_run(source_path=str(file_path))
        summary = BatchSummary()
        outcomes: list[BatchRecordOutcome] = []
        touched: set[tuple[str, str]] = set()

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    summary.total += 1
                    outcome, touch = self._process_line(line_number, line, batch_id=batch_id)
                    outcomes.append(outcome)
                    self._accumulate(summary, outcome)
                    if touch:
                        touched.add(touch)

            for user_id, score_date in sorted(touched):
                refresh_daily_user_score(
                    self.repo,
                    user_id=user_id,
                    score_date=score_date,
                    config=self.fusion_config,
                )

            status = "completed"
            if summary.failed > 0 and summary.processed == 0 and summary.skipped == 0:
                status = "failed"
            self.repo.complete_batch_run(
                batch_id,
                status=status,
                total_records=summary.total,
                succeeded=summary.processed,
                failed=summary.failed,
                invalid=summary.invalid,
                unsupported=summary.unsupported,
                skipped=summary.skipped,
            )
        except Exception:
            self.repo.complete_batch_run(
                batch_id,
                status="failed",
                total_records=summary.total,
                succeeded=summary.processed,
                failed=summary.failed,
                invalid=summary.invalid,
                unsupported=summary.unsupported,
                skipped=summary.skipped,
            )
            raise

        result = BatchProcessingResult(
            source=str(file_path),
            summary=summary,
            records=outcomes,
            batch_id=batch_id,
        )
        logger.info(
            "Persistent batch complete batch_id=%s total=%s processed=%s skipped=%s failed=%s",
            batch_id,
            summary.total,
            summary.processed,
            summary.skipped,
            summary.failed,
        )
        return result

    def _process_line(
        self,
        line_number: int,
        line: str,
        *,
        batch_id: str,
    ) -> tuple[BatchRecordOutcome, Optional[tuple[str, str]]]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return (
                BatchRecordOutcome(
                    line_number=line_number,
                    status="invalid",
                    error=f"malformed JSON: {exc.msg}",
                ),
                None,
            )
        if not isinstance(payload, dict):
            return (
                BatchRecordOutcome(
                    line_number=line_number,
                    status="invalid",
                    error="record must be a JSON object",
                ),
                None,
            )
        return self._process_payload(line_number, payload, batch_id=batch_id)

    def _process_payload(
        self,
        line_number: int,
        payload: dict[str, Any],
        *,
        batch_id: str,
    ) -> tuple[BatchRecordOutcome, Optional[tuple[str, str]]]:
        activity_id = payload.get("activity_id")
        user_id = payload.get("user_id")
        activity_type = payload.get("activity_type")
        id_hint = str(activity_id) if activity_id is not None else None
        user_hint = str(user_id) if user_id is not None else None
        type_hint = str(activity_type) if activity_type is not None else None

        try:
            activity = parse_activity_record(payload)
        except ValidationError as exc:
            return (
                BatchRecordOutcome(
                    line_number=line_number,
                    status="invalid",
                    activity_id=id_hint,
                    user_id=user_hint,
                    activity_type=type_hint,
                    error=_format_validation_error(exc),
                ),
                None,
            )
        except Exception as exc:  # noqa: BLE001
            return (
                BatchRecordOutcome(
                    line_number=line_number,
                    status="invalid",
                    activity_id=id_hint,
                    user_id=user_hint,
                    activity_type=type_hint,
                    error=str(exc),
                ),
                None,
            )

        # Idempotency: skip successful analyses; allow retry after failure.
        existing_status = self.repo.get_analysis_status(activity.activity_id)
        if existing_status == "processed":
            return (
                BatchRecordOutcome(
                    line_number=line_number,
                    status="skipped",
                    activity_id=activity.activity_id,
                    user_id=activity.user_id,
                    activity_type=activity.activity_type,
                    note=(
                        f"activity_id={activity.activity_id} already processed; "
                        "skipped to avoid silent duplication"
                    ),
                ),
                None,
            )

        if not self.repo.activity_exists(activity.activity_id):
            inserted = self.repo.insert_activity(activity, batch_id=batch_id)
            if not inserted:
                return (
                    BatchRecordOutcome(
                        line_number=line_number,
                        status="skipped",
                        activity_id=activity.activity_id,
                        user_id=activity.user_id,
                        activity_type=activity.activity_type,
                        note=(
                            f"activity_id={activity.activity_id} already stored; "
                            "skipped to avoid silent duplication"
                        ),
                    ),
                    None,
                )
        elif existing_status == "failed":
            logger.info(
                "Retrying previously failed activity_id=%s line=%s",
                activity.activity_id,
                line_number,
            )

        try:
            result = self.pipeline.analyze_activity(activity)
        except Exception as exc:  # noqa: BLE001 — isolate failures
            logger.exception(
                "Failed processing activity_id=%s line=%s",
                activity.activity_id,
                line_number,
            )
            score_date = self.repo.upsert_analysis_result(
                activity_id=activity.activity_id,
                user_id=activity.user_id,
                status="failed",
                batch_id=batch_id,
                error=str(exc),
                processed_at=activity.created_at,
            )
            return (
                BatchRecordOutcome(
                    line_number=line_number,
                    status="failed",
                    activity_id=activity.activity_id,
                    user_id=activity.user_id,
                    activity_type=activity.activity_type,
                    error=str(exc),
                ),
                (activity.user_id, score_date),
            )

        score_date = self.repo.upsert_analysis_result(
            activity_id=activity.activity_id,
            user_id=activity.user_id,
            status="processed",
            batch_id=batch_id,
            result=result,
        )
        return (
            BatchRecordOutcome(
                line_number=line_number,
                status="processed",
                activity_id=activity.activity_id,
                user_id=activity.user_id,
                activity_type=activity.activity_type,
                result=result,
            ),
            (activity.user_id, score_date),
        )

    @staticmethod
    def _accumulate(summary: BatchSummary, outcome: BatchRecordOutcome) -> None:
        if outcome.status == "invalid":
            summary.invalid += 1
            return
        if outcome.status == "skipped":
            summary.skipped += 1
            summary.valid += 1
            return
        summary.valid += 1
        if outcome.status == "processed":
            summary.processed += 1
        elif outcome.status == "unsupported":
            summary.unsupported += 1
        elif outcome.status == "failed":
            summary.failed += 1
