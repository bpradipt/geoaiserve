"""Common API routes for health checks and model information."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from ..config import get_settings
from ..models import registry
from ..schemas import (
    HealthResponse,
    ModelInfo,
    ModelType,
    ModelsListResponse,
)

router = APIRouter(tags=["common"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and list loaded models"
)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with service status and loaded models
    """
    settings = get_settings()
    loaded_models = registry.list_loaded_models()

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        models_loaded=[m["model_type"] for m in loaded_models]
    )


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List available models",
    description="Get information about all available model types and their configurations"
)
async def list_models() -> ModelsListResponse:
    """List all available models and their capabilities.

    Returns:
        ModelsListResponse containing model information
    """
    settings = get_settings()
    available_types = registry.get_available_model_types()
    loaded_models = {m["model_type"]: m for m in registry.list_loaded_models()}

    # Model descriptions and supported tasks
    model_info_map = {
        ModelType.SAM: {
            "description": "Segment Anything Model for image segmentation",
            "supported_tasks": ["automatic_mask_generation", "prompt_based_segmentation"],
            "default_name": settings.sam_model_name,
        },
        ModelType.MOONDREAM: {
            "description": "Vision-language model for VQA, captioning, and detection",
            "supported_tasks": ["image_captioning", "visual_qa", "object_detection", "point_detection"],
            "default_name": settings.moondream_model_name,
        },
        ModelType.DINOV3: {
            "description": "DINOv2 for feature extraction and similarity analysis",
            "supported_tasks": ["feature_extraction", "patch_similarity", "batch_similarity"],
            "default_name": settings.dinov3_model_name,
        },
        ModelType.GROUNDED_SAM: {
            "description": "Text-prompted segmentation with Grounded SAM",
            "supported_tasks": ["text_prompted_segmentation"],
            "default_name": "grounded-sam",
        },
        ModelType.DETECTRON2: {
            "description": "Instance segmentation with Detectron2",
            "supported_tasks": ["instance_segmentation"],
            "default_name": "detectron2",
        },
        ModelType.TIMM: {
            "description": "Classification and segmentation with TIMM models",
            "supported_tasks": ["image_classification", "segmentation"],
            "default_name": "timm",
        },
    }

    models = []
    for model_type in ModelType:
        # Skip if not in available types and not in config
        if model_type not in available_types and model_type.value not in settings.geoai_models:
            continue

        info = model_info_map.get(model_type, {
            "description": f"{model_type.value} model",
            "supported_tasks": [],
            "default_name": model_type.value,
        })

        loaded_info = loaded_models.get(model_type.value)
        is_loaded = loaded_info is not None

        models.append(
            ModelInfo(
                model_id=model_type.value,
                model_type=model_type,
                model_name=loaded_info["model_name"] if loaded_info else info["default_name"],
                description=info["description"],
                supported_tasks=info["supported_tasks"],
                device=loaded_info["device"] if loaded_info else settings.device,
                loaded=is_loaded,
            )
        )

    return ModelsListResponse(
        models=models,
        total=len(models)
    )


@router.get(
    "/models/{model_id}/info",
    response_model=ModelInfo,
    summary="Get model details",
    description="Get detailed information about a specific model"
)
async def get_model_info(model_id: str) -> ModelInfo:
    """Get detailed information about a specific model.

    Args:
        model_id: Model identifier

    Returns:
        ModelInfo for the requested model

    Raises:
        HTTPException: If model not found
    """
    try:
        model_type = ModelType(model_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found"
        )

    # Get all models and find the requested one
    all_models = await list_models()
    for model in all_models.models:
        if model.model_id == model_id:
            return model

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model '{model_id}' not found"
    )
