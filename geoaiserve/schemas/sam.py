"""SAM-specific request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import GeoMetadata
from .requests import BaseInferenceRequest


class SamGenerateRequest(BaseInferenceRequest):
    """Request for SAM automatic mask generation."""

    points_per_side: int = Field(
        default=32,
        ge=1,
        le=64,
        description="Number of points per side for grid sampling"
    )
    pred_iou_thresh: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="IoU threshold for filtering mask predictions"
    )
    stability_score_thresh: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Stability score threshold for filtering masks"
    )
    crop_n_layers: int = Field(
        default=0,
        ge=0,
        description="Number of crop layers for processing large images"
    )
    crop_n_points_downscale_factor: int = Field(
        default=1,
        ge=1,
        description="Downscale factor for points in crop layers"
    )
    min_mask_region_area: int = Field(
        default=0,
        ge=0,
        description="Minimum area for mask regions (pixels)"
    )


class SamPredictRequest(BaseInferenceRequest):
    """Request for SAM prompt-based segmentation."""

    point_coords: list[list[float]] | None = Field(
        None,
        description="Point coordinates [[x, y], ...] for prompting"
    )
    point_labels: list[int] | None = Field(
        None,
        description="Point labels (1=foreground, 0=background)"
    )
    boxes: list[list[float]] | None = Field(
        None,
        description="Bounding boxes [[x1, y1, x2, y2], ...] for prompting"
    )
    multimask_output: bool = Field(
        default=True,
        description="Whether to output multiple candidate masks"
    )


class SamBatchRequest(BaseModel):
    """Request for batch SAM processing."""

    file_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of file IDs to process"
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of images to process in parallel"
    )
    predict_params: SamPredictRequest = Field(
        default_factory=SamPredictRequest,
        description="Parameters for each prediction"
    )


class MaskResult(BaseModel):
    """Individual mask result."""

    mask_id: int = Field(..., description="Mask identifier")
    score: float = Field(..., description="Confidence score")
    area: int = Field(..., description="Mask area in pixels")
    bbox: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    geometry: dict | None = Field(None, description="Mask geometry as GeoJSON")


class SamGenerateResponse(BaseModel):
    """Response for SAM automatic mask generation."""

    status: str = Field(default="success", description="Response status")
    num_masks: int = Field(..., description="Number of generated masks")
    masks: list[MaskResult] = Field(..., description="Generated masks")
    metadata: GeoMetadata = Field(..., description="Processing metadata")
    download_url: str | None = Field(
        None,
        description="URL to download full results"
    )


class SamPredictResponse(BaseModel):
    """Response for SAM prompt-based segmentation."""

    status: str = Field(default="success", description="Response status")
    masks: list[MaskResult] = Field(..., description="Predicted masks")
    metadata: GeoMetadata = Field(..., description="Processing metadata")
    download_url: str | None = Field(
        None,
        description="URL to download results"
    )


class SamBatchResponse(BaseModel):
    """Response for batch SAM processing."""

    status: str = Field(default="success", description="Response status")
    total_images: int = Field(..., description="Total images processed")
    results: list[SamPredictResponse] = Field(..., description="Batch results")
    metadata: GeoMetadata = Field(..., description="Processing metadata")
