"""Moondream vision-language model API endpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from ..config import get_settings
from ..models import registry
from ..schemas.common import GeoMetadata, ModelType
from ..schemas.moondream import (
    DetectionResult,
    MoondreamCaptionRequest,
    MoondreamCaptionResponse,
    MoondreamDetectRequest,
    MoondreamDetectResponse,
    MoondreamPointRequest,
    MoondreamPointResponse,
    MoondreamQueryRequest,
    MoondreamQueryResponse,
    PointResult,
)
from ..services.file_handler import file_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/moondream", tags=["Moondream"])


def resolve_image_path(request) -> Path:
    """Resolve image path from request (file_id, URL, or base64).

    Args:
        request: Request with file_id or image input

    Returns:
        Path to the image file

    Raises:
        HTTPException: If no valid image source provided
    """
    if request.file_id:
        return file_handler.get_file_by_id(request.file_id)
    elif request.image and request.image.url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL input not supported in sync context. Please upload the file first."
        )
    elif request.image and request.image.base64:
        return file_handler.decode_base64(request.image.base64)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either file_id or image input must be provided"
        )


@router.post(
    "/caption",
    response_model=MoondreamCaptionResponse,
    summary="Image captioning",
    description="Generate descriptive captions for images"
)
async def generate_caption(
    request: MoondreamCaptionRequest,
) -> MoondreamCaptionResponse:
    """Generate image caption using Moondream.

    Args:
        request: Caption generation parameters including file_id

    Returns:
        MoondreamCaptionResponse with generated caption

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create Moondream model
        moondream_model = registry.get_model(
            model_type=ModelType.MOONDREAM,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Generate caption
        result = moondream_model.caption(
            image=image_path,
            length=request.length,
        )

        processing_time = time.time() - start_time

        return MoondreamCaptionResponse(
            status="success",
            caption=result["caption"],
            length=result["length"],
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=moondream_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Caption generation failed: {str(e)}"
        )


@router.post(
    "/query",
    response_model=MoondreamQueryResponse,
    summary="Visual question answering",
    description="Ask questions about images and get answers"
)
async def answer_question(
    request: MoondreamQueryRequest,
) -> MoondreamQueryResponse:
    """Answer questions about an image using Moondream.

    Args:
        request: Question and parameters including file_id

    Returns:
        MoondreamQueryResponse with answer

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create Moondream model
        moondream_model = registry.get_model(
            model_type=ModelType.MOONDREAM,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Answer question
        result = moondream_model.query(
            image=image_path,
            question=request.question,
        )

        processing_time = time.time() - start_time

        return MoondreamQueryResponse(
            status="success",
            question=result["question"],
            answer=result["answer"],
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=moondream_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual QA failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual QA failed: {str(e)}"
        )


@router.post(
    "/detect",
    response_model=MoondreamDetectResponse,
    summary="Object detection",
    description="Detect specific types of objects in images"
)
async def detect_objects(
    request: MoondreamDetectRequest,
) -> MoondreamDetectResponse:
    """Detect objects in an image using Moondream.

    Args:
        request: Detection parameters including file_id

    Returns:
        MoondreamDetectResponse with detected objects

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create Moondream model
        moondream_model = registry.get_model(
            model_type=ModelType.MOONDREAM,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Detect objects
        result = moondream_model.detect(
            image=image_path,
            object_type=request.object_type,
        )

        processing_time = time.time() - start_time

        # Parse detections
        detections = []
        for obj in result.get("detections", {}).get("objects", []):
            detections.append(
                DetectionResult(
                    label=obj.get("label", request.object_type),
                    confidence=1.0,
                    bbox=obj.get("bbox", [0, 0, 0, 0]),
                )
            )

        return MoondreamDetectResponse(
            status="success",
            object_type=request.object_type,
            detections=detections,
            num_detections=len(detections),
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=moondream_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Object detection failed: {str(e)}"
        )


@router.post(
    "/point",
    response_model=MoondreamPointResponse,
    summary="Point detection",
    description="Find and point to specific objects in images"
)
async def point_to_objects(
    request: MoondreamPointRequest,
) -> MoondreamPointResponse:
    """Point to objects in an image using Moondream.

    Args:
        request: Point detection parameters including file_id

    Returns:
        MoondreamPointResponse with point coordinates

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create Moondream model
        moondream_model = registry.get_model(
            model_type=ModelType.MOONDREAM,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Point to objects
        result = moondream_model.point(
            image=image_path,
            object_description=request.object_description,
        )

        processing_time = time.time() - start_time

        # Parse points
        points = []
        for pt in result.get("points", []):
            points.append(
                PointResult(
                    x=pt[0],
                    y=pt[1],
                    confidence=1.0,
                )
            )

        return MoondreamPointResponse(
            status="success",
            object_description=request.object_description,
            points=points,
            num_points=len(points),
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=moondream_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Point detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Point detection failed: {str(e)}"
        )
