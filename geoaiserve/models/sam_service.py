"""SAM (Segment Anything Model) service implementation using segment-geospatial."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..schemas.common import DeviceType, ModelType
from .base import BaseGeoModel

logger = logging.getLogger(__name__)


class SAMService(BaseGeoModel):
    """Service for SAM (Segment Anything Model) inference using SamGeo3.

    This service wraps segment-geospatial's SamGeo3 to provide geospatial
    segmentation capabilities with SAM 2.1 model.
    """

    def __init__(
        self,
        model_name: str = "sam2.1-hiera-large",
        device: DeviceType = DeviceType.CPU,
        **kwargs: Any,
    ):
        """Initialize SAM service.

        Args:
            model_name: SAM model variant (sam2.1-hiera-large, sam2.1-hiera-base, etc.)
            device: Device to run inference on
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, device, **kwargs)
        self._current_image_path: str | None = None

    @property
    def model_type(self) -> ModelType:
        """Return the model type."""
        return ModelType.SAM

    @property
    def supported_tasks(self) -> list[str]:
        """Return list of supported tasks."""
        return [
            "automatic_mask_generation",
            "prompt_based_segmentation",
            "batch_processing",
        ]

    def load(self) -> None:
        """Load the SAM model into memory."""
        if self._loaded:
            logger.info(f"SAM model already loaded: {self.model_name}")
            return

        # Use mock model if explicitly requested via GEOAI_ALLOW_MOCK
        if self._allow_mock:
            logger.info(
                "GEOAI_ALLOW_MOCK is set. Creating mock SAM service."
            )
            self._model = self._create_mock_model()
            self._is_mock = True
            self._loaded = True
            return

        try:
            logger.info(f"Loading SAM model: {self.model_name} on {self.device}")

            from samgeo.samgeo3 import SamGeo3

            # Initialize SamGeo3 with meta backend for interactive segmentation
            self._model = SamGeo3(
                model_id=self.model_name,
                backend="meta",
                enable_inst_interactivity=True,
            )
            self._loaded = True
            logger.info(f"SAM model loaded successfully: {self.model_name}")

        except ImportError as e:
            raise ImportError(
                "Required dependency 'segment-geospatial' not installed for SAM. "
                "Install with: pip install 'segment-geospatial[samgeo3]'"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise

    def _create_mock_model(self) -> Any:
        """Create a mock model for testing when actual model is not available."""

        class MockSAM:
            def __init__(self):
                self._image_set = False

            def set_image(self, source):
                logger.info(f"Mock SAM: set_image called with {source}")
                self._image_set = True

            def generate_masks_by_points_inst(
                self, point_coords=None, point_labels=None, **kwargs
            ):
                logger.info("Mock SAM: generate_masks_by_points_inst called")
                return {
                    "masks": np.array([[[1, 0], [0, 1]]]),
                    "scores": np.array([0.95]),
                }

            def generate_masks_by_boxes_inst(self, boxes=None, **kwargs):
                logger.info("Mock SAM: generate_masks_by_boxes_inst called")
                return {
                    "masks": np.array([[[1, 0], [0, 1]]]),
                    "scores": np.array([0.95]),
                }

            def generate(self, source, output=None, **kwargs):
                logger.info("Mock SAM: generate called")
                return {"status": "mock", "message": "SAM model not installed"}

            def save_masks(self, output, **kwargs):
                logger.info(f"Mock SAM: save_masks called with output={output}")

        return MockSAM()

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._loaded:
            logger.info(f"Unloading SAM model: {self.model_name}")
            self._model = None
            self._current_image_path = None
            self._loaded = False

    def _set_image_if_needed(self, image_path: Path | str) -> None:
        """Set image on the model if it's different from current."""
        image_path_str = str(image_path)
        if self._current_image_path != image_path_str:
            logger.info(f"Setting image: {image_path_str}")
            self._model.set_image(image_path_str)
            self._current_image_path = image_path_str

    def predict(
        self,
        image_path: Path | str,
        point_coords: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        boxes: list[list[float]] | None = None,
        multimask_output: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run prompt-based segmentation using SamGeo3.

        Args:
            image_path: Path to input image (GeoTIFF supported)
            point_coords: Point prompts [[x, y], ...]
            point_labels: Point labels [1=foreground, 0=background]
            boxes: Box prompts [[x1, y1, x2, y2], ...]
            multimask_output: Whether to output multiple masks
            **kwargs: Additional parameters

        Returns:
            Dictionary containing masks, scores, and metadata
        """
        if not self._loaded:
            self.load()

        try:
            logger.info(f"Running SAM prediction on {image_path}")

            # Set the image on the model
            self._set_image_if_needed(image_path)

            # Use box prompts if provided
            if boxes is not None:
                logger.info(f"Using box prompts: {boxes}")
                result = self._model.generate_masks_by_boxes_inst(
                    boxes=boxes,
                    multimask_output=multimask_output,
                    **kwargs,
                )
            # Use point prompts if provided
            elif point_coords is not None:
                logger.info(f"Using point prompts: {point_coords}")

                # Convert inputs to numpy arrays
                np_point_coords = np.array(point_coords)
                np_point_labels = (
                    np.array(point_labels)
                    if point_labels is not None
                    else np.ones(len(point_coords), dtype=np.int32)
                )

                result = self._model.generate_masks_by_points_inst(
                    point_coords=np_point_coords,
                    point_labels=np_point_labels,
                    multimask_output=multimask_output,
                    **kwargs,
                )
            else:
                raise ValueError("Either point_coords or boxes must be provided")

            # Handle different result formats
            if isinstance(result, dict):
                return {
                    "masks": result.get("masks"),
                    "scores": result.get("scores"),
                    "logits": result.get("logits"),
                }
            else:
                # If result is the masks directly
                return {
                    "masks": result,
                    "scores": None,
                    "logits": None,
                }

        except Exception as e:
            logger.error(f"SAM prediction failed: {e}")
            raise

    def generate_masks(
        self,
        image_path: Path | str,
        output_path: Path | str | None = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run automatic mask generation.

        Args:
            image_path: Path to input image
            output_path: Path to save output
            points_per_side: Number of points per side for sampling
            pred_iou_thresh: IoU threshold for mask prediction
            stability_score_thresh: Stability score threshold
            crop_n_layers: Number of crop layers
            crop_n_points_downscale_factor: Downscale factor for points
            min_mask_region_area: Minimum mask region area
            **kwargs: Additional parameters

        Returns:
            Dictionary containing generated masks and metadata
        """
        if not self._loaded:
            self.load()

        try:
            logger.info(f"Running SAM automatic mask generation on {image_path}")

            # Run automatic mask generation
            self._model.generate(
                source=str(image_path),
                output=str(output_path) if output_path else None,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
                **kwargs,
            )

            return {
                "status": "success",
                "output_path": str(output_path) if output_path else None,
                "params": {
                    "points_per_side": points_per_side,
                    "pred_iou_thresh": pred_iou_thresh,
                    "stability_score_thresh": stability_score_thresh,
                },
            }

        except Exception as e:
            logger.error(f"SAM mask generation failed: {e}")
            raise

    def predict_batch(
        self,
        image_paths: list[Path | str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run batch prediction on multiple images.

        Args:
            image_paths: List of image paths
            **kwargs: Additional parameters for predict()

        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, **kwargs)
                results.append({"image": str(image_path), "result": result})
            except Exception as e:
                logger.error(f"Batch prediction failed for {image_path}: {e}")
                results.append({"image": str(image_path), "error": str(e)})

        return results
