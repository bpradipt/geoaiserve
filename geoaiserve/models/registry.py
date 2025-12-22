"""Model registry for centralized model management with lazy loading."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ..config import get_settings
from ..schemas.common import DeviceType, ModelType
from .base import BaseGeoModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Centralized model management with lazy loading and caching."""

    _instance: ModelRegistry | None = None
    _models: dict[str, BaseGeoModel] = {}
    _model_classes: dict[ModelType, type[BaseGeoModel]] = {}

    def __new__(cls) -> ModelRegistry:
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_model_class(
        cls,
        model_type: ModelType,
        model_class: type[BaseGeoModel]
    ) -> None:
        """Register a model class for a specific model type.

        Args:
            model_type: Type of model (SAM, Moondream, etc.)
            model_class: Class implementing BaseGeoModel
        """
        cls._model_classes[model_type] = model_class
        logger.info(f"Registered model class {model_class.__name__} for {model_type}")

    def _generate_key(
        self,
        model_type: ModelType,
        model_name: str,
        device: DeviceType,
        **kwargs: Any,
    ) -> str:
        """Generate unique cache key for model instance.

        Args:
            model_type: Type of model
            model_name: Model name or path
            device: Device to run on
            **kwargs: Additional parameters

        Returns:
            Unique cache key
        """
        # Create deterministic key from parameters
        key_parts = [
            model_type.value,
            model_name,
            device.value,
            str(sorted(kwargs.items())),
        ]
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
        return f"{model_type.value}_{key_hash}"

    def get_model(
        self,
        model_type: ModelType,
        model_name: str | None = None,
        device: DeviceType | None = None,
        auto_load: bool = True,
        **kwargs: Any,
    ) -> BaseGeoModel:
        """Get or create a model instance with caching.

        Args:
            model_type: Type of model to get
            model_name: Optional model name (uses config default if not provided)
            device: Optional device (uses config default if not provided)
            auto_load: Whether to automatically load the model
            **kwargs: Additional model-specific parameters

        Returns:
            Model instance

        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in self._model_classes:
            raise ValueError(
                f"Model type {model_type} not registered. "
                f"Available types: {list(self._model_classes.keys())}"
            )

        settings = get_settings()

        # Use defaults from config if not provided
        if model_name is None:
            model_name = self._get_default_model_name(model_type)
        if device is None:
            device = DeviceType(settings.device)

        # Generate cache key
        cache_key = self._generate_key(model_type, model_name, device, **kwargs)

        # Return cached model if available
        if cache_key in self._models:
            logger.debug(f"Returning cached model: {cache_key}")
            return self._models[cache_key]

        # Create new model instance
        logger.info(
            f"Creating new {model_type} model: "
            f"name={model_name}, device={device}"
        )
        model_class = self._model_classes[model_type]
        model = model_class(
            model_name=model_name,
            device=device,
            **kwargs,
        )

        # Load model if requested
        if auto_load:
            logger.info(f"Loading model: {cache_key}")
            model.load()

        # Cache the model
        self._models[cache_key] = model
        return model

    def _get_default_model_name(self, model_type: ModelType) -> str:
        """Get default model name from configuration.

        Args:
            model_type: Type of model

        Returns:
            Default model name from config

        Raises:
            ValueError: If no default configured for model type
        """
        settings = get_settings()
        model_name_map = {
            ModelType.SAM: settings.sam_model_name,
            ModelType.MOONDREAM: settings.moondream_model_name,
            ModelType.DINOV3: settings.dinov3_model_name,
        }

        if model_type not in model_name_map:
            raise ValueError(f"No default model name configured for {model_type}")

        return model_name_map[model_type]

    def list_loaded_models(self) -> list[dict[str, Any]]:
        """List all currently loaded models.

        Returns:
            List of model information dictionaries
        """
        return [
            {
                "cache_key": key,
                "model_type": model.model_type.value,
                "model_name": model.model_name,
                "device": model.device.value,
                "loaded": model.loaded,
            }
            for key, model in self._models.items()
        ]

    def unload_model(self, cache_key: str) -> bool:
        """Unload a specific model from memory.

        Args:
            cache_key: Cache key of model to unload

        Returns:
            True if model was unloaded, False if not found
        """
        if cache_key in self._models:
            model = self._models[cache_key]
            logger.info(f"Unloading model: {cache_key}")
            model.unload()
            del self._models[cache_key]
            return True
        return False

    def unload_all(self) -> None:
        """Unload all models from memory."""
        logger.info("Unloading all models")
        for model in self._models.values():
            model.unload()
        self._models.clear()

    def get_available_model_types(self) -> list[ModelType]:
        """Get list of registered model types.

        Returns:
            List of available model types
        """
        return list(self._model_classes.keys())


# Global registry instance
registry = ModelRegistry()
