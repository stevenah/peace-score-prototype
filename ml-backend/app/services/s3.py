"""S3 upload helper — streams video files from disk to S3 without buffering."""

from __future__ import annotations

import logging

import boto3
from botocore.config import Config

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None and settings.s3_bucket:
        _client = boto3.client(
            "s3",
            region_name=settings.s3_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=Config(
                s3={"multipart_threshold": 8 * 1024 * 1024},  # 8MB
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
