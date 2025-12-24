"""Moondream vision-language model API endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, File, HTTPException, UploadFile, status

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
from ..services import file_handler

router = APIRouter(prefix="/moondream", tags=["Moondream"])


@router.post(
    "/caption",
    response_model=MoondreamCaptionResponse,
    summary="Image captioning",
    description="Generate descriptive captions for images"
)
async def generate_caption(
    file: UploadFile = File(..., description="Input image file"),
    request: MoondreamCaptionRequest = MoondreamCaptionRequest(),
) -> MoondreamCaptionResponse:
    """Generate image caption using Moondream.

    Args:
        file: Uploaded image file
        request: Caption generation parameters

    Returns:
        MoondreamCaptionResponse with generated caption

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Caption generation failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)


@router.post(
    "/query",
    response_model=MoondreamQueryResponse,
    summary="Visual question answering",
    description="Ask questions about images and get answers"
)
async def answer_question(
    file: UploadFile = File(..., description="Input image file"),
    request: MoondreamQueryRequest = MoondreamQueryRequest(question="What is in this image?"),
) -> MoondreamQueryResponse:
    """Answer questions about an image using Moondream.

    Args:
        file: Uploaded image file
        request: Question and parameters

    Returns:
        MoondreamQueryResponse with answer

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual QA failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)


@router.post(
    "/detect",
    response_model=MoondreamDetectResponse,
    summary="Object detection",
    description="Detect specific types of objects in images"
)
async def detect_objects(
    file: UploadFile = File(..., description="Input image file"),
    request: MoondreamDetectRequest = MoondreamDetectRequest(object_type="car"),
) -> MoondreamDetectResponse:
    """Detect objects in an image using Moondream.

    Args:
        file: Uploaded image file
        request: Detection parameters

    Returns:
        MoondreamDetectResponse with detected objects

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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
                    confidence=1.0,  # Mock confidence
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Object detection failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)


@router.post(
    "/point",
    response_model=MoondreamPointResponse,
    summary="Point detection",
    description="Find and point to specific objects in images"
)
async def point_to_objects(
    file: UploadFile = File(..., description="Input image file"),
    request: MoondreamPointRequest = MoondreamPointRequest(object_description="the main subject"),
) -> MoondreamPointResponse:
    """Point to objects in an image using Moondream.

    Args:
        file: Uploaded image file
        request: Point detection parameters

    Returns:
        MoondreamPointResponse with point coordinates

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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
                    confidence=1.0,  # Mock confidence
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Point detection failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)
