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

    # --- ESGE landmark stations (app/ml/landmarks) ---
    # Off by default. Shadow mode = enabled but not displayed: the backend
    # computes and logs stations, the UI shows nothing (fail-safe default).
    landmarks_enabled: bool = False
    landmarks_display: bool = False
    # Committed pin {version, s3_key, sha256} of the private-S3 bundle tarball.
    landmark_bundle_lock: str = str(_APP_DIR / "models" / "landmarks" / "BUNDLE.lock")
    # Where fetched bundles are unpacked (the Fly volume); falls back to a temp
    # dir when not writable (e.g. local dev).
    landmark_cache_dir: str = "/data/models/landmarks"
    # Local dev/tests only: serve this unpacked bundle dir, bypassing the lock.
    landmark_bundle_dir: str = ""
    # Classify every Nth live frame (skipped frames do not enter the tracker).
    landmark_every_n: int = 1

    # --- Serving resources ---
    torch_threads: int = 2
    max_live_sockets: int = 4
    # Load and run one dummy forward of each real model at startup, before the
    # batch worker starts (no-op with mock models).
    warmup_models: bool = True

    model_config = {"env_prefix": "PEACE_"}

    def public_summary(self) -> dict:
        """Effective settings for the startup log, without credentials."""
        hidden = {"aws_access_key_id", "aws_secret_access_key"}
        out = {k: v for k, v in self.model_dump().items() if k not in hidden}
        out["aws_credentials_set"] = bool(self.aws_access_key_id and self.aws_secret_access_key)
        return out


settings = Settings()
