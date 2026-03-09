from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PEACE ML Backend"
    version: str = "0.1.0"
    debug: bool = True
    use_mock_models: bool = True
    sample_rate_fps: float = 2.0
    max_upload_size_mb: int = 1024
    upload_dir: str = "/tmp/peace-uploads"
    cors_origins: list[str] = ["http://localhost:3000", "https://peace-frontend.fly.dev", "https://demo.gipeace.com"]
    model_path: str = ""
    device: str = "auto"
    job_db_path: str = "/tmp/peace-jobs/jobs.db"
    worker_poll_interval: float = 1.0

    # S3 config for video storage (optional — skipped if s3_bucket is empty)
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    model_config = {"env_prefix": "PEACE_"}


settings = Settings()
