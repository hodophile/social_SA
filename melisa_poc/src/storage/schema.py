"""SQLite DDL for MyUni POC persistence (not production schema)."""

from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batch_runs (
    batch_id TEXT PRIMARY KEY,
    source_path TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    total_records INTEGER NOT NULL DEFAULT 0,
    succeeded INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    invalid INTEGER NOT NULL DEFAULT 0,
    unsupported INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS activities (
    activity_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    text TEXT,
    media_path TEXT,
    created_at TEXT,
    metadata_json TEXT,
    batch_id TEXT,
    ingested_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batch_runs(batch_id)
);

CREATE INDEX IF NOT EXISTS idx_activities_user ON activities(user_id);

CREATE TABLE IF NOT EXISTS analysis_results (
    activity_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    overall_label TEXT,
    overall_score REAL,
    overall_confidence REAL,
    modalities_json TEXT,
    fusion_json TEXT,
    models_json TEXT,
    warnings_json TEXT,
    status TEXT NOT NULL,
    error TEXT,
    processed_at TEXT NOT NULL,
    score_date TEXT,
    batch_id TEXT,
    result_json TEXT,
    FOREIGN KEY (activity_id) REFERENCES activities(activity_id),
    FOREIGN KEY (batch_id) REFERENCES batch_runs(batch_id)
);

CREATE INDEX IF NOT EXISTS idx_analysis_user_date
    ON analysis_results(user_id, score_date);

CREATE TABLE IF NOT EXISTS daily_user_scores (
    user_id TEXT NOT NULL,
    score_date TEXT NOT NULL,
    activity_count INTEGER NOT NULL DEFAULT 0,
    valid_analysis_count INTEGER NOT NULL DEFAULT 0,
    mean_sentiment_score REAL,
    positive_count INTEGER NOT NULL DEFAULT 0,
    neutral_count INTEGER NOT NULL DEFAULT 0,
    negative_count INTEGER NOT NULL DEFAULT 0,
    daily_sentiment_label TEXT,
    updated_at TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT 'POC daily aggregate — not client business score',
    PRIMARY KEY (user_id, score_date)
);
"""
