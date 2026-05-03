"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_env_list(value: str) -> list[str]:
    """Parse .env list values: JSON array, comma-separated, '*', or empty."""
    s = (value or "").strip()
    if not s:
        return []
    if s == "*":
        return ["*"]
    if s.startswith("["):
        return json.loads(s)
    return [x.strip() for x in s.split(",") if x.strip()]


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Configuration
    app_name: str = "GeoAI REST API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # CORS Configuration (str so .env can use `*` or comma-separated; not JSON)
    enable_cors: bool = True
    cors_origins: str = "*"
    cors_credentials: bool = True
    cors_methods: str = "*"
    cors_headers: str = "*"

    # Model Configuration
    geoai_models: str = "sam,moondream,dinov3"
    # When true, DINOv3/SAM/Moondream use in-process mocks if torch/geoai are missing.
    # Set via env GEOAI_ALLOW_MOCK=1 or in .env (loaded by Settings).
    geoai_allow_mock: bool = False
    device: Literal["cuda", "cpu", "mps"] = "cpu"
    gpu_memory_limit: str = "8GB"

    # Model Paths (HuggingFace model identifiers)
    sam_model_name: str = "facebook/sam-vit-huge"
    moondream_model_name: str = "vikhyatk/moondream2"
    dinov3_model_name: str = "dinov3_vitl16"

    # Storage Configuration
    storage_backend: Literal["local", "s3", "gcs"] = "local"
    storage_path: Path = Path("/tmp/geoaiserve")
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    gcs_bucket: str | None = None

    # File Upload Configuration
    upload_dir: Path = Path("/tmp/geoaiserve/uploads")
    upload_ttl_hours: int = Field(default=24, ge=1, description="Hours before uploaded files are cleaned up")
    max_upload_size: int = 100 * 1024 * 1024  # 100 MB
    allowed_image_formats: str = (
        "image/tiff,image/jpeg,image/png,image/geotiff,application/octet-stream"
    )

    # Cache Configuration
    enable_cache: bool = False
    redis_url: str | None = None
    cache_ttl: int = 3600  # seconds

    # Job Queue Configuration
    enable_async_jobs: bool = False
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Security Configuration
    api_key_required: bool = False
    api_keys: str = ""
    secret_key: str = "your-secret-key-change-in-production"
    rate_limit: str = "100/minute"

    # Concurrency Configuration
    max_concurrent_inference: int = 1

    # Monitoring Configuration
    enable_metrics: bool = False
    metrics_port: int = 9090
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    # Feature Store Configuration
    feature_store_backend: Literal["zarr", "memory"] = "zarr"
    feature_store_path: Path = Path("/tmp/geoaiserve/features")

    @computed_field
    @property
    def cors_origins_list(self) -> list[str]:
        return _parse_env_list(self.cors_origins)

    @computed_field
    @property
    def cors_methods_list(self) -> list[str]:
        return _parse_env_list(self.cors_methods)

    @computed_field
    @property
    def cors_headers_list(self) -> list[str]:
        return _parse_env_list(self.cors_headers)

    @computed_field
    @property
    def geoai_models_list(self) -> list[str]:
        return _parse_env_list(self.geoai_models)

    @computed_field
    @property
    def api_keys_list(self) -> list[str]:
        return _parse_env_list(self.api_keys)

    @computed_field
    @property
    def allowed_image_formats_list(self) -> list[str]:
        return _parse_env_list(self.allowed_image_formats)

    def __init__(self, **kwargs):
        """Initialize settings and create storage directories."""
        super().__init__(**kwargs)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.feature_store_path.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
