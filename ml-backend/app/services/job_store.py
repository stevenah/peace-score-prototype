"""SQLite-backed job store for persistent video analysis tasks."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.api.schemas import AnalysisStatus
from app.config import settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Persistent job store backed by SQLite.

    Uses WAL mode for concurrent reads during writes and thread-local
    connections to safely serve both API handlers and the worker thread.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.job_db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'queued',
                progress REAL NOT NULL DEFAULT 0.0,
                file_path TEXT,
                video_metadata TEXT,
                results TEXT,
                error TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()

    def create_job(self, file_path: str) -> str:
        """Insert a new job and return its ID."""
        job_id = str(uuid.uuid4())
        now = _now_iso()
        conn = self._get_connection()
        conn.execute(
            """INSERT INTO jobs (id, status, file_path, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, AnalysisStatus.QUEUED.value, file_path, now, now),
        )
        conn.commit()
        return job_id

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Read a single job, returning a dict matching AnalysisResponse shape."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return self._row_to_response(row)

    def claim_next_job(self) -> Optional[dict[str, Any]]:
        """Atomically claim the oldest queued job for processing."""
        now = _now_iso()
        conn = self._get_connection()
        cursor = conn.execute(
            """UPDATE jobs
               SET status = ?, started_at = ?, updated_at = ?
               WHERE id = (
                   SELECT id FROM jobs
                   WHERE status = ?
                   ORDER BY created_at ASC
                   LIMIT 1
               )
               RETURNING *""",
            (AnalysisStatus.PROCESSING.value, now, now, AnalysisStatus.QUEUED.value),
        )
        row = cursor.fetchone()
        conn.commit()
        if row:
            return dict(row)
        return None

    def update_progress(self, job_id: str, progress: float) -> None:
        conn = self._get_connection()
        conn.execute(
            "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
            (progress, _now_iso(), job_id),
        )
        conn.commit()

    def complete_job(
        self, job_id: str, results: dict, metadata: dict
    ) -> None:
        """Mark job as completed with serialized results and metadata."""
        now = _now_iso()
        conn = self._get_connection()
        conn.execute(
            """UPDATE jobs
               SET status = ?, progress = 1.0,
                   results = ?, video_metadata = ?,
                   completed_at = ?, updated_at = ?
               WHERE id = ?""",
            (
                AnalysisStatus.COMPLETED.value,
                json.dumps(results),
                json.dumps(metadata),
                now,
                now,
                job_id,
            ),
        )
        conn.commit()

    def fail_job(self, job_id: str, error: str) -> None:
        """Fail a job. Requeues if retries remain, otherwise marks as failed."""
        now = _now_iso()
        conn = self._get_connection()
        conn.execute(
            """UPDATE jobs
               SET retry_count = retry_count + 1,
                   status = CASE
                       WHEN retry_count + 1 < max_retries THEN ?
                       ELSE ?
                   END,
                   error = ?,
                   updated_at = ?
               WHERE id = ?""",
            (
                AnalysisStatus.QUEUED.value,
                AnalysisStatus.FAILED.value,
                error,
                now,
                job_id,
            ),
        )
        conn.commit()

    def recover_stale_jobs(self) -> int:
        """Reset jobs stuck in 'processing' back to 'queued' (crash recovery)."""
        now = _now_iso()
        conn = self._get_connection()
        cursor = conn.execute(
            """UPDATE jobs
               SET status = ?, retry_count = retry_count + 1, updated_at = ?
               WHERE status = ? AND retry_count < max_retries""",
            (AnalysisStatus.QUEUED.value, now, AnalysisStatus.PROCESSING.value),
        )
        conn.commit()
        return cursor.rowcount

    def _row_to_response(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a DB row to the AnalysisResponse-compatible dict."""
        return {
            "analysis_id": row["id"],
            "status": row["status"],
            "progress": row["progress"],
            "video_metadata": json.loads(row["video_metadata"])
            if row["video_metadata"]
            else None,
            "results": json.loads(row["results"]) if row["results"] else None,
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "error": row["error"],
        }


# Singleton instance
job_store = JobStore()
