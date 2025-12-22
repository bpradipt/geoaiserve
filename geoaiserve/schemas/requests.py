"""Request schemas for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl

from .common import ModelConfig, OutputConfig


class ImageInput(BaseModel):
    """Base image input configuration."""

    url: HttpUrl | None = Field(None, description="URL to fetch image from")
    base64: str | None = Field(None, description="Base64 encoded image data")
    storage_path: str | None = Field(
        None,
        description="Path in cloud storage (s3://bucket/key or gs://bucket/key)"
    )

    def has_input(self) -> bool:
        """Check if any input method is specified."""
        return any([self.url, self.base64, self.storage_path])


class BaseInferenceRequest(BaseModel):
    """Base request for model inference."""

    image: ImageInput | None = Field(
        None,
        description="Image input configuration (alternative to file upload)"
    )
    model_params: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration"
    )
    async_processing: bool = Field(
        default=False,
        description="Whether to process asynchronously"
    )
