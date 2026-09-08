"""JSONL batch ingestion for MyUni daily-style activity workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from src.pipeline import MyUniSentimentPipeline
from src.schemas import (
    ActivityInput,
    BatchProcessingResult,
    BatchRecordOutcome,
    BatchSummary,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())) or "record"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) if parts else str(exc)


def parse_activity_record(raw: dict[str, Any]) -> ActivityInput:
    """Validate a single activity dict into ActivityInput."""
    return ActivityInput.model_validate(raw)


class BatchIngestor:
    """Read JSONL activity files, validate per record, and process what is supported."""

    def __init__(self, pipeline: Optional[MyUniSentimentPipeline] = None) -> None:
        self._pipeline = pipeline or MyUniSentimentPipeline()

    @property
    def pipeline(self) -> MyUniSentimentPipeline:
        return self._pipeline

    def process_file(self, path: PathLike) -> BatchProcessingResult:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Batch file not found: {file_path}")

        summary = BatchSummary()
        outcomes: list[BatchRecordOutcome] = []

        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                summary.total += 1
                outcome = self._process_line(line_number, line)
                outcomes.append(outcome)
                self._accumulate(summary, outcome)

        logger.info(
            "Batch complete source=%s total=%s valid=%s invalid=%s "
            "processed=%s unsupported=%s failed=%s",
            file_path,
            summary.total,
            summary.valid,
            summary.invalid,
            summary.processed,
            summary.unsupported,
            summary.failed,
        )
        return BatchProcessingResult(
            source=str(file_path),
            summary=summary,
            records=outcomes,
        )

    def process_records(
        self,
        records: list[dict[str, Any]],
        *,
        source: str = "<memory>",
    ) -> BatchProcessingResult:
        """Process an in-memory list of activity dicts (useful for tests)."""
        summary = BatchSummary()
        outcomes: list[BatchRecordOutcome] = []

        for line_number, raw in enumerate(records, start=1):
            summary.total += 1
            if not isinstance(raw, dict):
                outcome = BatchRecordOutcome(
                    line_number=line_number,
                    status="invalid",
                    error="record must be a JSON object",
                )
            else:
                outcome = self._process_payload(line_number, raw)
            outcomes.append(outcome)
            self._accumulate(summary, outcome)

        return BatchProcessingResult(
            source=source,
            summary=summary,
            records=outcomes,
        )

    def _process_line(self, line_number: int, line: str) -> BatchRecordOutcome:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return BatchRecordOutcome(
                line_number=line_number,
                status="invalid",
                error=f"malformed JSON: {exc.msg}",
            )

        if not isinstance(payload, dict):
            return BatchRecordOutcome(
                line_number=line_number,
                status="invalid",
                error="record must be a JSON object",
            )

        return self._process_payload(line_number, payload)

    def _process_payload(
        self,
        line_number: int,
        payload: dict[str, Any],
    ) -> BatchRecordOutcome:
        activity_id = payload.get("activity_id")
        user_id = payload.get("user_id")
        activity_type = payload.get("activity_type")
        id_hint = str(activity_id) if activity_id is not None else None
        user_hint = str(user_id) if user_id is not None else None
        type_hint = str(activity_type) if activity_type is not None else None

        try:
            activity = parse_activity_record(payload)
        except ValidationError as exc:
            return BatchRecordOutcome(
                line_number=line_number,
                status="invalid",
                activity_id=id_hint,
                user_id=user_hint,
                activity_type=type_hint,
                error=_format_validation_error(exc),
            )
        except Exception as exc:  # noqa: BLE001 — isolate batch records
            return BatchRecordOutcome(
                line_number=line_number,
                status="invalid",
                activity_id=id_hint,
                user_id=user_hint,
                activity_type=type_hint,
                error=str(exc),
            )

        try:
            result = self._pipeline.analyze_activity(activity)
        except Exception as exc:  # noqa: BLE001 — do not abort the batch
            logger.exception(
                "Failed processing activity_id=%s line=%s",
                activity.activity_id,
                line_number,
            )
            return BatchRecordOutcome(
                line_number=line_number,
                status="failed",
                activity_id=activity.activity_id,
                user_id=activity.user_id,
                activity_type=activity.activity_type,
                error=str(exc),
            )

        return BatchRecordOutcome(
            line_number=line_number,
            status="processed",
            activity_id=activity.activity_id,
            user_id=activity.user_id,
            activity_type=activity.activity_type,
            result=result,
        )

    @staticmethod
    def _accumulate(summary: BatchSummary, outcome: BatchRecordOutcome) -> None:
        if outcome.status == "invalid":
            summary.invalid += 1
            return

        summary.valid += 1
        if outcome.status == "processed":
            summary.processed += 1
        elif outcome.status == "unsupported":
            summary.unsupported += 1
        elif outcome.status == "failed":
            summary.failed += 1
