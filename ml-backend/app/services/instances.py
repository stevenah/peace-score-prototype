"""Shared singleton instances for the application."""

from app.services.job_store import job_store
from app.services.worker import AnalysisWorker

worker = AnalysisWorker(job_store)
