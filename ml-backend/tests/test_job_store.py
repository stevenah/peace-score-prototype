"""Tests for the SQLite-backed JobStore."""

from __future__ import annotations

from app.api.schemas import AnalysisStatus
from app.services.job_store import JobStore


class TestJobStoreCreate:
    def test_create_returns_uuid(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        assert len(job_id) == 36  # UUID format

    def test_created_job_is_queued(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        job = store.get_job(job_id)
        assert job is not None
        assert job["status"] == AnalysisStatus.QUEUED.value
        assert job["progress"] == 0.0

    def test_created_job_has_timestamps(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        job = store.get_job(job_id)
        assert job["created_at"]
        assert job["completed_at"] is None


class TestJobStoreGet:
    def test_get_nonexistent_returns_none(self, store: JobStore):
        assert store.get_job("no-such-id") is None

    def test_get_returns_response_shape(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        job = store.get_job(job_id)
        expected_keys = {
            "analysis_id", "status", "progress", "video_metadata",
            "results", "created_at", "completed_at", "error",
        }
        assert expected_keys.issubset(job.keys())


class TestJobStoreClaim:
    def test_claim_returns_oldest_queued(self, store: JobStore):
        id1 = store.create_job("/tmp/a.mp4")
        store.create_job("/tmp/b.mp4")
        claimed = store.claim_next_job()
        assert claimed is not None
        assert claimed["id"] == id1

    def test_claim_sets_processing(self, store: JobStore):
        job_id = store.create_job("/tmp/a.mp4")
        store.claim_next_job()
        job = store.get_job(job_id)
        assert job["status"] == AnalysisStatus.PROCESSING.value

    def test_claim_empty_returns_none(self, store: JobStore):
        assert store.claim_next_job() is None

    def test_claim_skips_processing_jobs(self, store: JobStore):
        store.create_job("/tmp/a.mp4")
        store.claim_next_job()  # claims first job
        assert store.claim_next_job() is None  # no more queued jobs


class TestJobStoreProgress:
    def test_update_progress(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        store.update_progress(job_id, 0.5)
        job = store.get_job(job_id)
        assert job["progress"] == 0.5


class TestJobStoreComplete:
    def test_complete_sets_status_and_results(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        results = {"peace_scores": {"overall": {"score": 2}}}
        metadata = {"duration_seconds": 30.0, "fps": 30.0}
        store.complete_job(job_id, results=results, metadata=metadata)

        job = store.get_job(job_id)
        assert job["status"] == AnalysisStatus.COMPLETED.value
        assert job["progress"] == 1.0
        assert job["results"] == results
        assert job["video_metadata"] == metadata
        assert job["completed_at"] is not None


class TestJobStoreFail:
    def test_fail_requeues_with_retries(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        store.fail_job(job_id, "crash")
        job = store.get_job(job_id)
        # First failure should requeue (retry_count=1 < max_retries=3)
        assert job["status"] == AnalysisStatus.QUEUED.value

    def test_fail_exhausted_retries_marks_failed(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        store.fail_job(job_id, "err1")
        store.fail_job(job_id, "err2")
        store.fail_job(job_id, "err3")
        job = store.get_job(job_id)
        assert job["status"] == AnalysisStatus.FAILED.value
        assert job["error"] == "err3"


class TestJobStoreRecovery:
    def test_recover_stale_jobs(self, store: JobStore):
        job_id = store.create_job("/tmp/test.mp4")
        store.claim_next_job()  # sets to processing
        recovered = store.recover_stale_jobs()
        assert recovered == 1
        job = store.get_job(job_id)
        assert job["status"] == AnalysisStatus.QUEUED.value

    def test_recover_does_not_touch_queued_or_completed(self, store: JobStore):
        id1 = store.create_job("/tmp/a.mp4")  # queued
        id2 = store.create_job("/tmp/b.mp4")
        store.complete_job(id2, results={}, metadata={})  # completed
        recovered = store.recover_stale_jobs()
        assert recovered == 0
