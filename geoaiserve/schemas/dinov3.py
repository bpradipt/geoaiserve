"""DINOv3-specific request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import GeoMetadata, ModelConfig
from .requests import BaseInferenceRequest


class DINOv3FeaturesRequest(BaseInferenceRequest):
    """Request for feature extraction."""

    return_patch_features: bool = Field(
        default=False,
        description="Whether to return patch-level features"
    )


class DINOv3SimilarityRequest(BaseInferenceRequest):
    """Request for patch similarity computation."""

    query_points: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Query points for similarity [[x, y], ...]"
    )


class DINOv3BatchSimilarityRequest(BaseModel):
    """Request for batch similarity computation."""

    query_file_id: str = Field(
        ...,
        description="File ID of the query image"
    )
    candidate_file_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of file IDs for candidate images"
    )
    model_params: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of top similar images to return"
    )


class SimilarityResult(BaseModel):
    """Similarity result for a candidate."""

    index: int = Field(..., description="Candidate index")
    similarity: float = Field(..., description="Similarity score (0-1)")
    error: str | None = Field(None, description="Error if computation failed")


class DINOv3FeaturesResponse(BaseModel):
    """Response for feature extraction."""

    status: str = Field(default="success", description="Response status")
    cls_token: list[float] = Field(..., description="CLS token features")
    patch_features: list[list[float]] | None = Field(
        None,
        description="Patch-level features if requested"
    )
    feature_dim: int = Field(..., description="Feature dimensionality")
    patch_grid: list[int] | None = Field(
        None,
        description="Patch grid dimensions [height, width]"
    )
    metadata: GeoMetadata = Field(..., description="Processing metadata")


class DINOv3SimilarityResponse(BaseModel):
    """Response for patch similarity."""

    status: str = Field(default="success", description="Response status")
    query_points: list[list[float]] = Field(..., description="Query points")
    similarity_maps: list[list[list[float]]] = Field(
        ...,
        description="Similarity heatmaps for each query point"
    )
    map_size: list[int] = Field(..., description="Similarity map dimensions [h, w]")
    metadata: GeoMetadata = Field(..., description="Processing metadata")
    download_url: str | None = Field(
        None,
        description="URL to download similarity maps as GeoTIFF"
    )


class DINOv3BatchSimilarityResponse(BaseModel):
    """Response for batch similarity."""

    status: str = Field(default="success", description="Response status")
    num_candidates: int = Field(..., description="Number of candidate images")
    similarities: list[SimilarityResult] = Field(
        ...,
        description="Similarity scores for each candidate (sorted)"
    )
    metadata: GeoMetadata = Field(..., description="Processing metadata")
