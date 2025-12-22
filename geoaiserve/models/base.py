"""Base model interface for all GeoAI models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas.common import DeviceType, ModelType


class BaseGeoModel(ABC):
    """Abstract base class for all GeoAI models."""

    def __init__(
        self,
        model_name: str,
        device: DeviceType = DeviceType.CPU,
        **kwargs: Any,
    ):
        """Initialize the model.

        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to run inference on (cuda, cpu, mps)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        self._model: Any = None
        self._loaded = False

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the model type."""
        pass

    @property
    @abstractmethod
    def supported_tasks(self) -> list[str]:
        """Return list of supported tasks."""
        pass

    @property
    def loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def predict(self, **kwargs: Any) -> dict[str, Any]:
        """Run inference on the model.

        Args:
            **kwargs: Task-specific parameters

        Returns:
            Dictionary containing inference results
        """
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"device={self.device}, "
            f"loaded={self.loaded})"
        )
