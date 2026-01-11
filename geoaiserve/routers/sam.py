"""SAM (Segment Anything Model) API endpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from ..config import get_settings
from ..models import registry
from ..schemas.common import GeoMetadata, ModelType
from ..schemas.sam import (
    SamBatchRequest,
    SamBatchResponse,
    SamGenerateRequest,
    SamGenerateResponse,
    SamPredictRequest,
    SamPredictResponse,
)
from ..services.file_handler import file_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sam", tags=["SAM"])


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
        # This would need async handling - for now raise error
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
    "/generate",
    response_model=SamGenerateResponse,
    summary="Automatic mask generation",
    description="Generate segmentation masks automatically using SAM's grid-based approach"
)
async def generate_masks(
    request: SamGenerateRequest,
) -> SamGenerateResponse:
    """Generate masks automatically using SAM.

    Args:
        request: Generation parameters including file_id

    Returns:
        SamGenerateResponse with generated masks

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()
    image_path = None

    try:
        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create SAM model
        sam_model = registry.get_model(
            model_type=ModelType.SAM,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Generate output path
        output_path = settings.storage_path / f"sam_output_{image_path.stem}.geojson"

        # Run mask generation
        result = sam_model.generate_masks(
            image_path=image_path,
            output_path=output_path,
            points_per_side=request.points_per_side,
            pred_iou_thresh=request.pred_iou_thresh,
            stability_score_thresh=request.stability_score_thresh,
            crop_n_layers=request.crop_n_layers,
            crop_n_points_downscale_factor=request.crop_n_points_downscale_factor,
            min_mask_region_area=request.min_mask_region_area,
        )

        processing_time = time.time() - start_time

        return SamGenerateResponse(
            status="success",
            num_masks=0,
            masks=[],
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=sam_model.model_name,
                output_crs=request.output.crs,
            ),
            download_url=f"/downloads/{output_path.name}" if output_path.exists() else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mask generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mask generation failed: {str(e)}"
        )


@router.post(
    "/predict",
    response_model=SamPredictResponse,
    summary="Prompt-based segmentation",
    description="Segment specific objects using point, box, or mask prompts"
)
async def predict_masks(
    request: SamPredictRequest,
) -> SamPredictResponse:
    """Predict masks using prompts (points, boxes).

    Args:
        request: Prediction parameters with prompts and file_id

    Returns:
        SamPredictResponse with predicted masks

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Validate prompts
        if not any([request.point_coords, request.boxes]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one prompt type (points or boxes) must be provided"
            )

        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create SAM model
        sam_model = registry.get_model(
            model_type=ModelType.SAM,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Run prediction
        result = sam_model.predict(
            image_path=image_path,
            point_coords=request.point_coords,
            point_labels=request.point_labels,
            boxes=request.boxes,
            multimask_output=request.multimask_output,
        )

        processing_time = time.time() - start_time

        return SamPredictResponse(
            status="success",
            masks=[],
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=sam_model.model_name,
                output_crs=request.output.crs,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=SamBatchResponse,
    summary="Batch processing",
    description="Process multiple images in batch mode"
)
async def batch_process(
    request: SamBatchRequest,
) -> SamBatchResponse:
    """Process multiple images in batch.

    Args:
        request: Batch processing parameters with file_ids

    Returns:
        SamBatchResponse with all results

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    if len(request.file_ids) > request.batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum batch size: {request.batch_size}"
        )

    try:
        # Resolve all file_ids to paths
        image_paths = []
        for file_id in request.file_ids:
            image_path = file_handler.get_file_by_id(file_id)
            image_paths.append(image_path)

        # Get or create SAM model
        sam_model = registry.get_model(
            model_type=ModelType.SAM,
            model_name=request.predict_params.model_params.model_name,
            device=request.predict_params.model_params.device,
        )

        # Run batch prediction
        batch_results = sam_model.predict_batch(
            image_paths=image_paths,
            point_coords=request.predict_params.point_coords,
            point_labels=request.predict_params.point_labels,
            boxes=request.predict_params.boxes,
            multimask_output=request.predict_params.multimask_output,
        )

        processing_time = time.time() - start_time

        # Build response
        results = []
        for batch_result in batch_results:
            if "error" not in batch_result:
                results.append(
                    SamPredictResponse(
                        status="success",
                        masks=[],
                        metadata=GeoMetadata(
                            processing_time=processing_time / len(request.file_ids),
                            model_name=sam_model.model_name,
                        ),
                    )
                )

        return SamBatchResponse(
            status="success",
            total_images=len(request.file_ids),
            results=results,
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=sam_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )
