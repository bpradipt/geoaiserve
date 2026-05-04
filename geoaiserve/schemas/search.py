"""Schemas for object / template search API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .common import ModelConfig


SearchOperation = Literal["template_search", "image_chat"]


class SourceRoiPct(BaseModel):
    """Source region of interest in percent of image (0–100), matching spatialint-ui area mode."""

    x: float = Field(..., ge=0, le=100)
    y: float = Field(..., ge=0, le=100)
    width: float = Field(..., ge=0, le=100)
    height: float = Field(..., ge=0, le=100)


class SearchOptions(BaseModel):
    """Tuning knobs for multi-scale template search."""

    multi_scale: bool = Field(default=True, description="Enable multi-scale query/target processing")
    max_query_side: int = Field(default=512, ge=64, le=2048)
    max_target_side: int = Field(default=896, ge=128, le=2048)
    query_scale_factors: list[float] = Field(
        default_factory=lambda: [1.0, 0.75, 0.5, 0.35],
        description="Relative scales applied to max_query_side for the source template",
    )
    target_scale_factors: list[float] = Field(
        default_factory=lambda: [1.0, 0.65],
        description="Relative scales applied to max_target_side for the target image",
    )
    min_peak_distance_patches: int = Field(
        default=2,
        ge=1,
        description="Minimum patch-grid distance between peaks (NMS)",
    )
    iou_nms_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    """Unified search request (decoupled from frontend filenames)."""

    operation: SearchOperation = Field(
        default="template_search",
        description=(
            "template_search: DINOv3 template match (requires target file id(s)). "
            "image_chat: vision-language chat about source_file_id (spatialint-ui mock-compatible)."
        ),
    )
    chat_message: str | None = Field(
        None,
        description="User message for operation=image_chat (required for that mode).",
    )
    conversation_history: list[dict[str, Any]] | None = Field(
        None,
        description="Optional prior turns for image_chat (reserved; Moondream uses current message).",
    )
    source_file_id: str = Field(..., min_length=1)
    target_file_id: str | None = Field(
        None,
        description="Single target file id (use this or target_file_ids)",
    )
    target_file_ids: list[str] | None = Field(
        None,
        min_length=1,
        description="Multiple target file ids",
    )
    source_roi: SourceRoiPct | None = Field(
        None,
        description="Optional ROI on source (percent). If omitted, whole source image is used.",
    )
    top_k_matches_per_target: int = Field(default=5, ge=1, le=50)
    top_k_targets: int | None = Field(
        None,
        ge=1,
        le=100,
        description="If set with multiple targets, keep only this many targets ranked by best match score",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity (0–1) for a detection",
    )
    strategy: Literal["auto", "dinov3_multiscale_template_v1"] = Field(
        default="auto",
        description="Search strategy; auto uses dinov3_multiscale_template_v1",
    )
    model_params: ModelConfig = Field(default_factory=ModelConfig)
    options: SearchOptions = Field(default_factory=SearchOptions)

    @model_validator(mode="after")
    def _one_target_mode(self) -> SearchRequest:
        if self.operation == "image_chat":
            if self.chat_message is None or not str(self.chat_message).strip():
                raise ValueError("chat_message is required when operation is image_chat")
            return self
        has_single = self.target_file_id is not None and self.target_file_id != ""
        has_multi = self.target_file_ids is not None and len(self.target_file_ids) > 0
        if has_single and has_multi:
            raise ValueError("Provide either target_file_id or target_file_ids, not both")
        if not has_single and not has_multi:
            raise ValueError("Provide target_file_id or target_file_ids")
        return self


class BboxPx(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class BboxPct(BaseModel):
    x: float
    y: float
    width: float
    height: float


class SearchMatch(BaseModel):
    score: float = Field(..., description="Cosine similarity in [0, 1]")
    confidence: int = Field(..., ge=0, le=100, description="UI-friendly score 0–100")
    bbox_px: BboxPx
    bbox_pct: BboxPct
    meta: dict[str, Any] = Field(default_factory=dict, description="Optional debug: scales, etc.")


class PerTargetSearchResult(BaseModel):
    target_file_id: str
    image_width: int
    image_height: int
    matches: list[SearchMatch]
    warnings: list[str] = Field(default_factory=list)


class SearchSummary(BaseModel):
    num_targets_requested: int
    num_targets_returned: int
    total_matches: int
    processing_time_ms: int


class SearchResponse(BaseModel):
    status: str = "success"
    strategy_used: str
    source_file_id: str
    mode: Literal["single", "multiple"]
    # Image chat (spatialint-ui /mock parity); omitted for template_search
    success: bool | None = None
    response: str | None = Field(
        None,
        description="Assistant reply when operation was image_chat",
    )
    selections: list[dict[str, Any]] | None = Field(
        None,
        description="Overlay boxes in percent 0–100: x, y, width, height, label",
    )
    clearSelections: bool | None = None
    # Mock-compatible single-target fields (percent boxes, confidence 0–100)
    totalMatches: int | None = None
    avgConfidence: int | None = None
    searchTime: int | None = None
    matches: list[dict[str, Any]] | None = None
    # Mock-compatible multi-target fields
    sourceImage: str | None = None
    targetImages: list[str] | None = None
    results: list[dict[str, Any]] | None = None
    summary: dict[str, Any] | None = None
    # Full structured results
    targets: list[PerTargetSearchResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
