"""Object / template search API (multi-scale DINOv3 POC)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import require_inference_slot
from ..schemas.search import SearchRequest, SearchResponse
from ..services.search_service import run_image_chat, run_search

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["Search"],
    dependencies=[Depends(require_inference_slot)],
)


@router.post(
    "/",
    response_model=SearchResponse,
    summary="Template search or image chat",
    description=(
        "Template search: match a source image (or ROI) against one or more targets "
        "using multi-scale DINOv3 patch similarity. "
        "Image chat: set operation to image_chat with chat_message and source_file_id "
        "for Moondream Q&A plus optional highlight boxes (spatialint-ui mock parity)."
    ),
)
def search_endpoint(request: SearchRequest) -> SearchResponse:
    """Run search or image chat; blocks until complete (POC)."""
    try:
        if request.operation == "image_chat":
            return run_image_chat(request)
        return run_search(request)
    except ImportError as exc:
        # Typical: missing optional `geoai` / ML stack when GEOAI_ALLOW_MOCK is unset.
        msg = str(exc).lower()
        if "geoai" in msg or "dinov3" in msg or "moondream" in msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "This endpoint requires the ML stack (geoai / models). "
                    "Install with `uv sync --group ml`, or set `GEOAI_ALLOW_MOCK=1` "
                    "for demo mode."
                ),
            ) from exc
        raise
