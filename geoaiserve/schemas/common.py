"""Common Pydantic schemas for requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    """Supported output formats for geospatial data."""

    JSON = "json"
    GEOJSON = "geojson"
    GEOTIFF = "geotiff"
    SHAPEFILE = "shapefile"
    GEOPACKAGE = "geopackage"
    KML = "kml"


class DeviceType(str, Enum):
    """Supported device types for model inference."""

    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class JobStatus(str, Enum):
    """Job processing status."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(str, Enum):
    """Available model types."""

    SAM = "sam"
    MOONDREAM = "moondream"
    DINOV3 = "dinov3"
    GROUNDED_SAM = "grounded-sam"
    DETECTRON2 = "detectron2"
    TIMM = "timm"
    AUTO = "auto"


class GeoMetadata(BaseModel):
    """Geospatial metadata for responses."""

    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: str = Field(..., description="Model used for processing")
    input_crs: str | None = Field(None, description="Input coordinate reference system")
    output_crs: str | None = Field(None, description="Output coordinate reference system")
    pixel_size: tuple[float, float] | None = Field(
        None, description="Pixel size (x, y) in CRS units"
    )
    bounds: list[float] | None = Field(
        None, description="Bounding box [minx, miny, maxx, maxy]"
    )
    transform: list[float] | None = Field(
        None, description="Affine transform coefficients"
    )


class OutputConfig(BaseModel):
    """Output configuration for geospatial results."""

    format: OutputFormat = Field(
        default=OutputFormat.GEOJSON,
        description="Output format for results"
    )
    simplify: float | None = Field(
        None,
        ge=0,
        description="Simplification tolerance for vector outputs"
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Output coordinate reference system"
    )


class ModelConfig(BaseModel):
    """Model configuration for inference."""

    model_name: str | None = Field(
        None,
        description="HuggingFace model identifier or local path"
    )
    device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Device to run inference on"
    )
    batch_size: int | None = Field(
        None,
        ge=1,
        description="Batch size for processing"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server timestamp"
    )
    models_loaded: list[str] = Field(
        default_factory=list,
        description="List of currently loaded models"
    )


class ModelInfo(BaseModel):
    """Information about an available model."""

    model_id: str = Field(..., description="Unique model identifier")
    model_type: ModelType = Field(..., description="Type of model")
    model_name: str = Field(..., description="HuggingFace or local model name")
    description: str = Field(..., description="Model description")
    supported_tasks: list[str] = Field(..., description="List of supported tasks")
    device: str = Field(..., description="Device model is loaded on")
    loaded: bool = Field(..., description="Whether model is currently loaded")


class ModelsListResponse(BaseModel):
    """Response containing list of available models."""

    models: list[ModelInfo] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of models")


class JobResponse(BaseModel):
    """Response for async job submission."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str | None = Field(None, description="Status message")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation timestamp"
    )


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str | None = Field(None, description="Status or error message")
    progress: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Progress percentage"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(None, description="Job start timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    result_url: str | None = Field(
        None,
        description="URL to download results if completed"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Any | None = Field(None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )


class GeoResponse(BaseModel):
    """Standard geospatial response."""

    status: str = Field(default="success", description="Response status")
    result: dict[str, Any] | list[Any] = Field(..., description="Result data")
    metadata: GeoMetadata = Field(..., description="Processing metadata")
    download_url: str | None = Field(
        None,
        description="URL to download result file"
    )
