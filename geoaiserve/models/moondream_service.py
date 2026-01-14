"""Moondream vision-language model service implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from ..schemas.common import DeviceType, ModelType
from .base import BaseGeoModel

logger = logging.getLogger(__name__)


class MoondreamService(BaseGeoModel):
    """Service for Moondream vision-language model inference."""

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
        self.revision = kwargs.get("revision", "2024-08-26")

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
        """Load the Moondream model into memory."""
        if self._loaded:
            logger.info(f"Moondream model already loaded: {self.model_name}")
            return

        try:
            logger.info(f"Loading Moondream model: {self.model_name} on {self.device}")

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Load model and tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    revision=self.revision,
                )

                # Move to device
                if self.device == DeviceType.CUDA:
                    self._model = self._model.to("cuda")
                elif self.device == DeviceType.MPS:
                    self._model = self._model.to("mps")

                self._loaded = True
                logger.info(f"Moondream model loaded successfully: {self.model_name}")

            except ImportError as e:
                if self._allow_mock:
                    logger.warning(
                        "transformers not installed. Creating mock Moondream service. "
                        "Set allow_mock=False or unset GEOAI_ALLOW_MOCK to require real model."
                    )
                    self._model = self._create_mock_model()
                    self._tokenizer = None
                    self._is_mock = True
                    self._loaded = True
                else:
                    raise ImportError(
                        f"Required dependency 'transformers' not installed for Moondream. "
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
            self._tokenizer = None
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
        """Generate image caption.

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

            logger.info(f"Generating {length} caption")

            # Generate caption based on length
            if hasattr(self._model, "caption"):
                # Real model requires tokenizer and list of images
                if self._tokenizer is not None:
                    captions = self._model.caption([image], self._tokenizer, length=length)
                    caption = captions[0] if captions else ""
                else:
                    caption = self._model.caption(image, length=length)
            else:
                # Fallback for actual transformers model
                caption = "Image caption placeholder"

            return {
                "caption": caption,
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
        """Answer questions about the image.

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

            logger.info(f"Answering question: {question}")

            if hasattr(self._model, "query"):
                # Real model requires tokenizer
                if self._tokenizer is not None:
                    answer = self._model.query(image, self._tokenizer, question)
                else:
                    answer = self._model.query(image, question)
            else:
                answer = f"Answer to: {question}"

            return {
                "question": question,
                "answer": answer,
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
        """Detect objects in the image.

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

            logger.info(f"Detecting objects: {object_type}")

            if hasattr(self._model, "detect"):
                # Real model signature: detect(image, query, tokenizer)
                if self._tokenizer is not None:
                    detections = self._model.detect(image, object_type, self._tokenizer)
                else:
                    detections = self._model.detect(image, object_type)
                # Handle None result (no detections found)
                if detections is None:
                    detections = {"objects": []}
            else:
                detections = {"objects": []}

            return {
                "object_type": object_type,
                "detections": detections,
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
        """Point to objects in the image.

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

            logger.info(f"Pointing to: {object_description}")

            if hasattr(self._model, "point"):
                # Real model signature: point(image, query, tokenizer)
                if self._tokenizer is not None:
                    points = self._model.point(image, object_description, self._tokenizer)
                else:
                    points = self._model.point(image, object_description)
                # Handle None result
                if points is None:
                    points = []
            else:
                # Model doesn't support point detection
                logger.warning(
                    f"Model {self.model_name} does not support point detection. "
                    "This feature may require a different model revision."
                )
                points = []

            return {
                "object_description": object_description,
                "points": points,
            }

        except Exception as e:
            logger.error(f"Point detection failed: {e}")
            raise
