"""S3 helpers — upload and download video files to/from S3."""

from __future__ import annotations

import logging
import os

import boto3
from botocore.config import Config

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None and settings.s3_bucket:
        # PEACE_S3_ENDPOINT points at an S3-compatible server (e.g. MinIO on
        # http://localhost:9000) for fully local development. Unset => real AWS.
        endpoint = settings.s3_endpoint or None
        _client = boto3.client(
            "s3",
            region_name=settings.s3_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            endpoint_url=endpoint,
            config=Config(
                s3={
                    "multipart_threshold": 8 * 1024 * 1024,  # 8MB
                    **({"addressing_style": "path"} if endpoint else {}),
                },
            ),
        )
    return _client


def upload_video_to_s3(local_path: str, s3_key: str, content_type: str = "video/mp4") -> bool:
    """Upload a video file to S3 using multipart streaming from disk.

    Returns True on success, False if S3 is not configured or upload fails.
    """
    if not settings.s3_bucket:
        return False

    client = _get_client()
    if not client:
        return False

    try:
        client.upload_file(
            local_path,
            settings.s3_bucket,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("Uploaded %s to s3://%s/%s", local_path, settings.s3_bucket, s3_key)
        return True
    except Exception:
        logger.exception("Failed to upload %s to S3", local_path)
        return False


def download_from_s3(s3_key: str, local_path: str) -> bool:
    """Download a file from S3 to a local path.

    Returns True on success, False if S3 is not configured or download fails.
    """
    if not settings.s3_bucket:
        return False

    client = _get_client()
    if not client:
        return False

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(settings.s3_bucket, s3_key, local_path)
        logger.info("Downloaded s3://%s/%s to %s", settings.s3_bucket, s3_key, local_path)
        return True
    except Exception:
        logger.exception("Failed to download s3://%s/%s", settings.s3_bucket, s3_key)
        return False
