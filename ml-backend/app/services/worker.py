"""Dedicated worker thread for processing video analysis jobs."""

from __future__ import annotations

import logging
import os
import threading
import time

from app.config import settings
from app.ml.frame_sampler import extract_frames
from app.ml.pipeline import create_pipeline
from app.services.job_store import JobStore
from app.services.s3 import download_from_s3

logger = logging.getLogger(__name__)


class AnalysisWorker:
    """Daemon thread that polls the job store and processes videos."""

    def __init__(self, job_store: JobStore) -> None:
        self._job_store = job_store
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._cleanup_orphaned_uploads()
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
        last_cleanup = time.monotonic()
        while not self._stop_event.is_set():
            job = self._job_store.claim_next_job()
            if job:
                self._process_job(job)
            else:
                self._stop_event.wait(timeout=settings.worker_poll_interval)

            # Periodic cleanup every 10 minutes
            if time.monotonic() - last_cleanup > 600:
                self._cleanup_orphaned_uploads()
                last_cleanup = time.monotonic()

    def _cleanup_orphaned_uploads(self) -> None:
        """Remove upload files older than 1 hour that aren't referenced by active jobs."""
        upload_dir = settings.upload_dir
        if not os.path.isdir(upload_dir):
            return

        cutoff = time.time() - 3600  # 1 hour
        removed = 0
        for name in os.listdir(upload_dir):
            path = os.path.join(upload_dir, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info("Cleaned up %d orphaned upload file(s)", removed)

    def _process_job(self, job: dict) -> None:
        job_id = job["id"]
        file_path = job.get("file_path")
        s3_key = job.get("s3_key")
        local_path = file_path  # May be None for S3 jobs

        logger.info("Processing job %s (file_path=%s, s3_key=%s)", job_id, file_path, s3_key)

        try:
            self._job_store.update_progress(job_id, 0.05)

            # Download from S3 if this is an S3-sourced job
            if not file_path and s3_key:
                os.makedirs(settings.upload_dir, exist_ok=True)
                ext = os.path.splitext(s3_key)[1] or ".mp4"
                local_path = os.path.join(settings.upload_dir, f"{job_id}{ext}")
                if not download_from_s3(s3_key, local_path):
                    raise RuntimeError(f"Failed to download s3://{s3_key}")
                logger.info("Downloaded S3 file to %s", local_path)

            if not local_path:
                raise RuntimeError("Job has no file_path and no s3_key")

            self._job_store.update_progress(job_id, 0.1)

            # Extract frames
            frames, metadata = extract_frames(
                local_path, sample_rate_fps=settings.sample_rate_fps
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
            # Clean up local file
            if local_path:
                try:
                    os.remove(local_path)
                except OSError:
                    pass
