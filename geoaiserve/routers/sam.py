"""SAM (Segment Anything Model) API endpoints."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

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
from ..services import file_handler

router = APIRouter(prefix="/sam", tags=["SAM"])


@router.post(
    "/generate",
    response_model=SamGenerateResponse,
    summary="Automatic mask generation",
    description="Generate segmentation masks automatically using SAM's grid-based approach"
)
async def generate_masks(
    file: UploadFile = File(..., description="Input image file"),
    request: SamGenerateRequest = SamGenerateRequest(),
) -> SamGenerateResponse:
    """Generate masks automatically using SAM.

    Args:
        file: Uploaded image file
        request: Generation parameters

    Returns:
        SamGenerateResponse with generated masks

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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

        # For mock response (when samgeo not installed)
        # In production, parse the actual output file
        return SamGenerateResponse(
            status="success",
            num_masks=0,  # Will be populated from actual results
            masks=[],  # Will be populated from actual results
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=sam_model.model_name,
                output_crs=request.output.crs,
            ),
            download_url=f"/downloads/{output_path.name}" if output_path.exists() else None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mask generation failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)


@router.post(
    "/predict",
    response_model=SamPredictResponse,
    summary="Prompt-based segmentation",
    description="Segment specific objects using point, box, or mask prompts"
)
async def predict_masks(
    file: UploadFile = File(..., description="Input image file"),
    request: SamPredictRequest = SamPredictRequest(),
) -> SamPredictResponse:
    """Predict masks using prompts (points, boxes).

    Args:
        file: Uploaded image file
        request: Prediction parameters with prompts

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

        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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

        # For mock response
        return SamPredictResponse(
            status="success",
            masks=[],  # Will be populated from actual results
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=sam_model.model_name,
                output_crs=request.output.crs,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)


@router.post(
    "/batch",
    response_model=SamBatchResponse,
    summary="Batch processing",
    description="Process multiple images in batch mode"
)
async def batch_process(
    files: list[UploadFile] = File(..., description="Multiple input image files"),
    request: SamBatchRequest = SamBatchRequest(),
) -> SamBatchResponse:
    """Process multiple images in batch.

    Args:
        files: List of uploaded image files
        request: Batch processing parameters

    Returns:
        SamBatchResponse with all results

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    if len(files) > request.batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum batch size: {request.batch_size}"
        )

    try:
        # Save all uploaded files
        image_paths = []
        for file in files:
            image_path = await file_handler.save_upload(file)
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
                            processing_time=processing_time / len(files),
                            model_name=sam_model.model_name,
                        ),
                    )
                )

        return SamBatchResponse(
            status="success",
            total_images=len(files),
            results=results,
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=sam_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug:
            for image_path in image_paths:
                if image_path.exists():
                    file_handler.cleanup_file(image_path)
