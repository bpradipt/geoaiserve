"""Moondream vision-language model service using geoai library."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from ..schemas.common import DeviceType, ModelType
from .base import BaseGeoModel

logger = logging.getLogger(__name__)


class MoondreamService(BaseGeoModel):
    """Service for Moondream vision-language model using geoai's MoondreamGeo.

    This service wraps geoai's MoondreamGeo to provide vision-language
    capabilities optimized for satellite and geospatial imagery.
    """

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        device: DeviceType = DeviceType.CPU,
        **kwargs: Any,
    ):
        """Initialize Moondream service.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, device, **kwargs)
        self.revision = kwargs.get("revision")
        self._processor = None

    @property
    def model_type(self) -> ModelType:
        """Return the model type."""
        return ModelType.MOONDREAM

    @property
    def supported_tasks(self) -> list[str]:
        """Return list of supported tasks."""
        return [
            "image_captioning",
            "visual_qa",
            "object_detection",
            "point_detection",
        ]

    def load(self) -> None:
        """Load the Moondream model using geoai."""
        if self._loaded:
            logger.info(f"Moondream model already loaded: {self.model_name}")
            return

        # Use mock model if explicitly requested via GEOAI_ALLOW_MOCK
        if self._allow_mock:
            logger.info(
                "GEOAI_ALLOW_MOCK is set. Creating mock Moondream service."
            )
            self._model = self._create_mock_model()
            self._processor = None
            self._is_mock = True
            self._loaded = True
            return

        try:
            logger.info(f"Loading Moondream model: {self.model_name} on {self.device}")

            from geoai import MoondreamGeo

            device_str = self.device.value if self.device else "cpu"
            self._processor = MoondreamGeo(
                model_name=self.model_name,
                revision=self.revision,
                device=device_str,
            )
            self._model = self._processor
            self._loaded = True
            logger.info(f"Moondream model loaded successfully via geoai: {self.model_name}")

        except ImportError as e:
            raise ImportError(
                f"Required dependency 'geoai' not installed for Moondream. "
                f"Install with: uv sync --group ml"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Moondream model: {e}")
            raise

    def _create_mock_model(self) -> Any:
        """Create a mock model for testing."""

        class MockMoondream:
            def caption(self, image, length="normal"):
                return "A mock image caption for testing"

            def query(self, image, question):
                return f"Mock answer to: {question}"

            def detect(self, image, object_type):
                return {"objects": [{"label": object_type, "bbox": [100, 100, 200, 200]}]}

        return MockMoondream()

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._loaded:
            logger.info(f"Unloading Moondream model: {self.model_name}")
            self._model = None
            self._processor = None
            self._loaded = False

    def predict(
        self,
        image_path: Path | str,
        task: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run inference based on task type.

        Args:
            image_path: Path to input image
            task: Task type (caption, query, detect, point)
            **kwargs: Task-specific parameters

        Returns:
            Dictionary containing task results
        """
        if not self._loaded:
            self.load()

        # Load image
        image = Image.open(image_path)

        if task == "caption":
            return self.caption(image, **kwargs)
        elif task == "query":
            return self.query(image, **kwargs)
        elif task == "detect":
            return self.detect(image, **kwargs)
        elif task == "point":
            return self.point(image, **kwargs)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def caption(
        self,
        image: Image.Image | Path | str,
        length: str = "normal",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate image caption using geoai.

        Args:
            image: PIL Image or path to image
            length: Caption length (short, normal, long)
            **kwargs: Additional parameters

        Returns:
            Dictionary containing caption text
        """
        if not self._loaded:
            self.load()

        try:
            if isinstance(image, (Path, str)):
                image = Image.open(image)

            logger.info(f"Generating {length} caption via geoai")

            if self._is_mock:
                caption = self._model.caption(image, length=length)
                return {
                    "caption": caption,
                    "length": length,
                }

            # Use geoai's caption method
            result = self._processor.caption(source=image, length=length)

            return {
                "caption": result.get("caption", ""),
                "length": length,
            }

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise

    def query(
        self,
        image: Image.Image | Path | str,
        question: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Answer questions about the image using geoai.

        Args:
            image: PIL Image or path to image
            question: Question to answer
            **kwargs: Additional parameters

        Returns:
            Dictionary containing answer text
        """
        if not self._loaded:
            self.load()

        try:
            if isinstance(image, (Path, str)):
                image = Image.open(image)

            logger.info(f"Answering question via geoai: {question}")

            if self._is_mock:
                answer = self._model.query(image, question)
                return {
                    "question": question,
                    "answer": answer,
                }

            # Use geoai's query method (note: question first, then source)
            result = self._processor.query(question=question, source=image)

            return {
                "question": question,
                "answer": result.get("answer", ""),
            }

        except Exception as e:
            logger.error(f"Visual QA failed: {e}")
            raise

    def detect(
        self,
        image: Image.Image | Path | str,
        object_type: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Detect objects in the image using geoai.

        Args:
            image: PIL Image or path to image
            object_type: Type of object to detect
            **kwargs: Additional parameters

        Returns:
            Dictionary containing detected objects
        """
        if not self._loaded:
            self.load()

        try:
            if isinstance(image, (Path, str)):
                image = Image.open(image)

            logger.info(f"Detecting objects via geoai: {object_type}")

            if self._is_mock:
                detections = self._model.detect(image, object_type)
                return {
                    "object_type": object_type,
                    "detections": detections,
                }

            # Use geoai's detect method
            result = self._processor.detect(source=image, object_type=object_type)

            return {
                "object_type": object_type,
                "detections": result.get("objects", []),
            }

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise

    def point(
        self,
        image: Image.Image | Path | str,
        object_description: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Point to objects in the image using geoai.

        Args:
            image: PIL Image or path to image
            object_description: Description of object to point to
            **kwargs: Additional parameters

        Returns:
            Dictionary containing point coordinates
        """
        if not self._loaded:
            self.load()

        try:
            if isinstance(image, (Path, str)):
                image = Image.open(image)

            logger.info(f"Pointing to via geoai: {object_description}")

            if self._is_mock:
                # Mock doesn't have point method, return empty
                return {
                    "object_description": object_description,
                    "points": [],
                }

            # Use geoai's point method
            result = self._processor.point(
                source=image,
                object_description=object_description,
            )

            return {
                "object_description": object_description,
                "points": result.get("points", []),
            }

        except Exception as e:
            logger.error(f"Point detection failed: {e}")
            raise
