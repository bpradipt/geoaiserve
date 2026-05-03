"""Object / template search API (multi-scale DINOv3 POC)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import require_inference_slot
from ..schemas.search import SearchRequest, SearchResponse
from ..services.search_service import run_search

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["Search"],
    dependencies=[Depends(require_inference_slot)],
)


@router.post(
    "/",
    response_model=SearchResponse,
    summary="Template search",
    description=(
        "Match a source image (or ROI) against one or more target images using "
        "multi-scale DINOv3 patch similarity. Returns bounding boxes in pixel and "
        "percent coordinates plus mock-compatible fields for spatialint-ui."
    ),
)
def search_endpoint(request: SearchRequest) -> SearchResponse:
    """Run search; blocks until complete (POC)."""
    try:
        return run_search(request)
    except ImportError as exc:
        # Typical: DINOv3Service.load() when optional `geoai` is not installed
        # (uv sync without --group ml) and GEOAI_ALLOW_MOCK is unset.
        msg = str(exc).lower()
        if "geoai" in msg or "dinov3" in msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Search requires the DINOv3 / geoai stack. Install ML deps: "
                    "`uv sync --group ml`, or set environment variable "
                    "`GEOAI_ALLOW_MOCK=1` for demo mode (synthetic matches)."
                ),
            ) from exc
        raise
