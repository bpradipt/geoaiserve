"""Moondream-specific request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import GeoMetadata
from .requests import BaseInferenceRequest


class MoondreamCaptionRequest(BaseInferenceRequest):
    """Request for image captioning."""

    length: str = Field(
        default="normal",
        description="Caption length: short, normal, or long"
    )
    reasoning: bool = Field(
        default=False,
        description="Include reasoning in caption"
    )


class MoondreamQueryRequest(BaseInferenceRequest):
    """Request for visual question answering."""

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the image"
    )


class MoondreamDetectRequest(BaseInferenceRequest):
    """Request for object detection."""

    object_type: str = Field(
        ...,
        min_length=1,
        description="Type of object to detect (e.g., 'car', 'person')"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections"
    )


class MoondreamPointRequest(BaseInferenceRequest):
    """Request for point detection."""

    object_description: str = Field(
        ...,
        min_length=1,
        description="Description of object to point to"
    )


class DetectionResult(BaseModel):
    """Single detection result."""

    label: str = Field(..., description="Object label")
    confidence: float = Field(..., description="Detection confidence score")
    bbox: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    geometry: dict | None = Field(None, description="Detection geometry as GeoJSON")


class PointResult(BaseModel):
    """Point detection result."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: float | None = Field(None, description="Point confidence")


class MoondreamCaptionResponse(BaseModel):
    """Response for image captioning."""

    status: str = Field(default="success", description="Response status")
    caption: str = Field(..., description="Generated caption")
    length: str = Field(..., description="Caption length type")
    metadata: GeoMetadata = Field(..., description="Processing metadata")


class MoondreamQueryResponse(BaseModel):
    """Response for visual question answering."""

    status: str = Field(default="success", description="Response status")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Answer to the question")
    metadata: GeoMetadata = Field(..., description="Processing metadata")


class MoondreamDetectResponse(BaseModel):
    """Response for object detection."""

    status: str = Field(default="success", description="Response status")
    object_type: str = Field(..., description="Queried object type")
    detections: list[DetectionResult] = Field(..., description="Detected objects")
    num_detections: int = Field(..., description="Number of detections")
    metadata: GeoMetadata = Field(..., description="Processing metadata")
    download_url: str | None = Field(
        None,
        description="URL to download results as GeoJSON"
    )


class MoondreamPointResponse(BaseModel):
    """Response for point detection."""

    status: str = Field(default="success", description="Response status")
    object_description: str = Field(..., description="Object description")
    points: list[PointResult] = Field(..., description="Detected points")
    num_points: int = Field(..., description="Number of points found")
    metadata: GeoMetadata = Field(..., description="Processing metadata")
    download_url: str | None = Field(
        None,
        description="URL to download results as GeoJSON"
    )
