"""DINOv3 feature extraction and similarity API endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, File, HTTPException, UploadFile, status

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
from ..services import file_handler

router = APIRouter(prefix="/dinov3", tags=["DINOv3"])


@router.post(
    "/features",
    response_model=DINOv3FeaturesResponse,
    summary="Feature extraction",
    description="Extract dense features from images using DINOv3"
)
async def extract_features(
    file: UploadFile = File(..., description="Input image file"),
    request: DINOv3FeaturesRequest = DINOv3FeaturesRequest(),
) -> DINOv3FeaturesResponse:
    """Extract features from an image using DINOv3.

    Args:
        file: Uploaded image file
        request: Feature extraction parameters

    Returns:
        DINOv3FeaturesResponse with extracted features

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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
            cls_token=result["cls_token"][0] if isinstance(result["cls_token"][0], list) else result["cls_token"],
            patch_features=result["patch_features"],
            feature_dim=result["feature_dim"],
            metadata=GeoMetadata(
                processing_time=processing_time,
                model_name=dinov3_model.model_name,
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)


@router.post(
    "/similarity",
    response_model=DINOv3SimilarityResponse,
    summary="Patch similarity",
    description="Compute patch-level similarity heatmaps for query points"
)
async def compute_similarity(
    file: UploadFile = File(..., description="Input image file"),
    request: DINOv3SimilarityRequest = DINOv3SimilarityRequest(query_points=[[100, 100]]),
) -> DINOv3SimilarityResponse:
    """Compute patch similarity for query points using DINOv3.

    Args:
        file: Uploaded image file
        request: Similarity computation parameters

    Returns:
        DINOv3SimilarityResponse with similarity maps

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save uploaded file
        image_path = await file_handler.save_upload(file)

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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity computation failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug and image_path.exists():
            file_handler.cleanup_file(image_path)


@router.post(
    "/batch-similarity",
    response_model=DINOv3BatchSimilarityResponse,
    summary="Batch similarity",
    description="Find most similar images from a batch of candidates"
)
async def batch_similarity(
    query_file: UploadFile = File(..., description="Query image file"),
    candidate_files: list[UploadFile] = File(..., description="Candidate image files"),
    request: DINOv3BatchSimilarityRequest = DINOv3BatchSimilarityRequest(),
) -> DINOv3BatchSimilarityResponse:
    """Compute similarity between query and multiple candidate images.

    Args:
        query_file: Query image file
        candidate_files: List of candidate image files
        request: Batch similarity parameters

    Returns:
        DINOv3BatchSimilarityResponse with ranked similarities

    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Save query image
        query_path = await file_handler.save_upload(query_file)

        # Save candidate images
        candidate_paths = []
        for candidate_file in candidate_files:
            candidate_path = await file_handler.save_upload(candidate_file)
            candidate_paths.append(candidate_path)

        # Get or create DINOv3 model
        dinov3_model = registry.get_model(
            model_type=ModelType.DINOV3,
            model_name=request.model_params.get("model_name"),
            device=request.model_params.get("device", "cpu"),
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch similarity failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files if needed
        if not settings.debug:
            if query_path.exists():
                file_handler.cleanup_file(query_path)
            for candidate_path in candidate_paths:
                if candidate_path.exists():
                    file_handler.cleanup_file(candidate_path)
