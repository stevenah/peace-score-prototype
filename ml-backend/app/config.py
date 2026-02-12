from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PEACE ML Backend"
    version: str = "0.1.0"
    debug: bool = True
    use_mock_models: bool = True
    sample_rate_fps: float = 2.0
    max_upload_size_mb: int = 500
    upload_dir: str = "/tmp/peace-uploads"
    cors_origins: list[str] = ["http://localhost:3000"]
    model_path: str = ""
    device: str = "auto"

    model_config = {"env_prefix": "PEACE_"}


settings = Settings()
