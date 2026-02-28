"""Dedicated worker thread for processing video analysis jobs."""

from __future__ import annotations

import logging
import os
import threading

from app.config import settings
from app.ml.frame_sampler import extract_frames
from app.ml.pipeline import create_pipeline
from app.services.job_store import JobStore

logger = logging.getLogger(__name__)


class AnalysisWorker:
    """Daemon thread that polls the job store and processes videos."""

    def __init__(self, job_store: JobStore) -> None:
        self._job_store = job_store
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Analysis worker started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Analysis worker stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            job = self._job_store.claim_next_job()
            if job:
                self._process_job(job)
            else:
                self._stop_event.wait(timeout=settings.worker_poll_interval)

    def _process_job(self, job: dict) -> None:
        job_id = job["id"]
        file_path = job["file_path"]
        logger.info("Processing job %s from %s", job_id, file_path)

        try:
            self._job_store.update_progress(job_id, 0.1)

            # Extract frames
            frames, metadata = extract_frames(
                file_path, sample_rate_fps=settings.sample_rate_fps
            )
            self._job_store.update_progress(job_id, 0.3)

            # Run pipeline
            pipeline = create_pipeline()
            frame_tuples = [(f.index, f.timestamp, f.data) for f in frames]
            results = pipeline.analyze_frames(frame_tuples, metadata["duration_seconds"])
            self._job_store.update_progress(job_id, 0.8)

            # Aggregate results
            aggregated = pipeline.aggregate_results(
                results, metadata["duration_seconds"]
            )
            self._job_store.update_progress(job_id, 0.95)

            # Complete
            self._job_store.complete_job(job_id, results=aggregated, metadata=metadata)
            logger.info("Job %s completed successfully", job_id)

        except Exception as e:
            logger.exception("Job %s failed: %s", job_id, e)
            self._job_store.fail_job(job_id, str(e))

        finally:
            # Clean up uploaded file only on final completion or exhausted retries
            job_after = self._job_store.get_job(job_id)
            if job_after and job_after["status"] in ("completed", "failed"):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
