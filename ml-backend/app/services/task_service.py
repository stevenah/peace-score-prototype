"""In-memory task management for async video analysis."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.api.schemas import AnalysisStatus


class TaskStore:
    """Simple in-memory store for analysis tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "analysis_id": task_id,
            "status": AnalysisStatus.QUEUED,
            "progress": 0.0,
            "video_metadata": None,
            "results": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        }
        return task_id

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs: Any) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].update(kwargs)

    def complete_task(self, task_id: str, results: dict, metadata: dict) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].update(
                {
                    "status": AnalysisStatus.COMPLETED,
                    "progress": 1.0,
                    "results": results,
                    "video_metadata": metadata,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def fail_task(self, task_id: str, error: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].update(
                {
                    "status": AnalysisStatus.FAILED,
                    "error": error,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )


# Singleton
task_store = TaskStore()
