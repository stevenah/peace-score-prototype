from pathlib import Path
from pydantic_settings import BaseSettings
import dotenv

_APP_DIR = Path(__file__).resolve().parent

dotenv.load_dotenv(_APP_DIR.parent / ".env")


class Settings(BaseSettings):
    app_name: str = "PEACE ML Backend"
    version: str = "0.1.0"
    debug: bool = True
    use_mock_models: bool = False
    sample_rate_fps: float = 2.0
    max_upload_size_mb: int = 1024
    upload_dir: str = "/tmp/peace-uploads"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://peace-frontend.fly.dev",
        "https://demo.gipeace.com",
    ]
    # Defaults to the model bundled in the repo so a fresh clone works with no config.
    model_path: str = str(_APP_DIR / "models" / "best_model.pt")
    device: str = "auto"
    job_db_path: str = "/tmp/peace-jobs/jobs.db"
    worker_poll_interval: float = 1.0

    # Disk space: reject uploads when free space drops below this threshold
    min_free_disk_mb: int = 512

    # S3 config for video storage (optional — skipped if s3_bucket is empty)
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    # Set to e.g. http://localhost:9000 to use MinIO instead of real AWS.
    s3_endpoint: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    model_config = {"env_prefix": "PEACE_"}


settings = Settings()
