"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # CORS Configuration
    enable_cors: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    cors_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_headers: list[str] = Field(default_factory=lambda: ["*"])

    # Model Configuration
    geoai_models: list[str] = Field(
        default_factory=lambda: ["sam", "moondream", "dinov3"]
    )
    device: Literal["cuda", "cpu", "mps"] = "cpu"
    gpu_memory_limit: str = "8GB"

    # Model Paths (HuggingFace model identifiers)
    sam_model_name: str = "facebook/sam-vit-huge"
    moondream_model_name: str = "vikhyatk/moondream2"
    dinov3_model_name: str = "facebook/dinov2-base"

    # Storage Configuration
    storage_backend: Literal["local", "s3", "gcs"] = "local"
    storage_path: Path = Path("/tmp/geoaiserve")
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    gcs_bucket: str | None = None

    # File Upload Configuration
    max_upload_size: int = 100 * 1024 * 1024  # 100 MB
    allowed_image_formats: list[str] = Field(
        default_factory=lambda: [
            "image/tiff",
            "image/jpeg",
            "image/png",
            "image/geotiff",
            "application/octet-stream",
        ]
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
    api_keys: list[str] = Field(default_factory=list)
    secret_key: str = "your-secret-key-change-in-production"
    rate_limit: str = "100/minute"

    # Monitoring Configuration
    enable_metrics: bool = False
    metrics_port: int = 9090
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    # Feature Store Configuration
    feature_store_backend: Literal["zarr", "memory"] = "zarr"
    feature_store_path: Path = Path("/tmp/geoaiserve/features")

    def __init__(self, **kwargs):
        """Initialize settings and create storage directories."""
        super().__init__(**kwargs)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.feature_store_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
