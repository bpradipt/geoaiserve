"""DINOv3 feature extraction and similarity API endpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from ..config import get_settings
from ..models import registry
from ..schemas.common import GeoMetadata, ModelType
from ..schemas.dinov3 import (
    DINOv3BatchSimilarityRequest,
    DINOv3BatchSimilarityResponse,
    DINOv3FeaturesRequest,
    DINOv3FeaturesResponse,
    DINOv3SimilarityRequest,
    DINOv3SimilarityResponse,
)
from ..services.file_handler import file_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dinov3", tags=["DINOv3"])


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
    "/features",
    response_model=DINOv3FeaturesResponse,
    summary="Feature extraction",
    description="Extract dense features from images using DINOv3"
)
async def extract_features(
    request: DINOv3FeaturesRequest,
) -> DINOv3FeaturesResponse:
    """Extract features from an image using DINOv3.

    Args:
        request: Feature extraction parameters including file_id

    Returns:
        DINOv3FeaturesResponse with extracted features

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create DINOv3 model
        dinov3_model = registry.get_model(
            model_type=ModelType.DINOV3,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Extract features
        result = dinov3_model.extract_features(
            image=image_path,
            return_patch_features=request.return_patch_features,
        )

        processing_time = time.time() - start_time

        return DINOv3FeaturesResponse(
            status="success",
            cls_token=result["cls_token"],
            patch_features=result["patch_features"],
            feature_dim=result["feature_dim"],
            patch_grid=result.get("patch_grid"),
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=dinov3_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}"
        )


@router.post(
    "/similarity",
    response_model=DINOv3SimilarityResponse,
    summary="Patch similarity",
    description="Compute patch-level similarity heatmaps for query points"
)
async def compute_similarity(
    request: DINOv3SimilarityRequest,
) -> DINOv3SimilarityResponse:
    """Compute patch similarity for query points using DINOv3.

    Args:
        request: Similarity computation parameters including file_id

    Returns:
        DINOv3SimilarityResponse with similarity maps

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Resolve image path from file_id
        image_path = resolve_image_path(request)

        # Get or create DINOv3 model
        dinov3_model = registry.get_model(
            model_type=ModelType.DINOV3,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Compute patch similarity
        result = dinov3_model.compute_patch_similarity(
            image=image_path,
            query_points=request.query_points,
        )

        processing_time = time.time() - start_time

        return DINOv3SimilarityResponse(
            status="success",
            query_points=result["query_points"],
            similarity_maps=result["similarity_maps"],
            map_size=result["map_size"],
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=dinov3_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity computation failed: {str(e)}"
        )


@router.post(
    "/batch-similarity",
    response_model=DINOv3BatchSimilarityResponse,
    summary="Batch similarity",
    description="Find most similar images from a batch of candidates"
)
async def batch_similarity(
    request: DINOv3BatchSimilarityRequest,
) -> DINOv3BatchSimilarityResponse:
    """Compute similarity between query and multiple candidate images.

    Args:
        request: Batch similarity parameters with query_file_id and candidate_file_ids

    Returns:
        DINOv3BatchSimilarityResponse with ranked similarities

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Resolve query image path
        query_path = file_handler.get_file_by_id(request.query_file_id)

        # Resolve candidate image paths
        candidate_paths = []
        for file_id in request.candidate_file_ids:
            candidate_path = file_handler.get_file_by_id(file_id)
            candidate_paths.append(candidate_path)

        # Get or create DINOv3 model
        dinov3_model = registry.get_model(
            model_type=ModelType.DINOV3,
            model_name=request.model_params.model_name,
            device=request.model_params.device,
        )

        # Compute batch similarity
        result = dinov3_model.batch_similarity(
            query_image=query_path,
            candidate_images=candidate_paths,
        )

        processing_time = time.time() - start_time

        # Limit to top_k results
        similarities = result["similarities"][:request.top_k]

        return DINOv3BatchSimilarityResponse(
            status="success",
            num_candidates=result["num_candidates"],
            similarities=similarities,
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=dinov3_model.model_name,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch similarity failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch similarity failed: {str(e)}"
        )
