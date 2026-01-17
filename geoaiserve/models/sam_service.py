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
            "text_prompted_segmentation",  # generate_masks(prompt="tree")
            "point_based_segmentation",    # predict(point_coords=...)
            "box_based_segmentation",      # predict(boxes=...) or generate_masks_by_boxes()
            "batch_processing",            # predict_batch()
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

            # Convert device to string format
            device_str = "cuda" if self.device == DeviceType.CUDA else "cpu"

            # Initialize SamGeo3 with meta backend for interactive segmentation
            self._model = SamGeo3(
                model_id=self.model_name,
                backend="meta",
                device=device_str,
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

            def set_image(self, source, bands=None):
                logger.info(f"Mock SAM: set_image called with {source}")
                self._image_set = True

            def predict_inst(
                self,
                point_coords=None,
                point_labels=None,
                box=None,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
                normalize_coords=True,
                point_crs=None,
                box_crs=None,
            ):
                """Mock predict_inst returning (masks, scores, logits) tuple."""
                logger.info("Mock SAM: predict_inst called")
                masks = np.array([[[1, 0], [0, 1]]])
                scores = np.array([0.95])
                logits = np.array([[[0.1, -0.1], [-0.1, 0.1]]])
                return masks, scores, logits

            def generate_masks_by_boxes_inst(
                self, boxes=None, box_crs=None, output=None, **kwargs
            ):
                """Mock generate_masks_by_boxes_inst returning dict."""
                logger.info("Mock SAM: generate_masks_by_boxes_inst called")
                return {
                    "masks": np.array([[[1, 0], [0, 1]]]),
                    "scores": np.array([0.95]),
                    "num_masks": 1,
                }

            def generate_masks(self, prompt, min_size=0, max_size=None, **kwargs):
                """Mock generate_masks for text-based segmentation."""
                logger.info(f"Mock SAM: generate_masks called with prompt={prompt}")
                return [{"mask": np.array([[1, 0], [0, 1]]), "score": 0.95}]

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
        multimask_output: bool = True,
        point_crs: str | None = None,
        box_crs: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run prompt-based segmentation using SamGeo3.

        Uses predict_inst for point/box prompts (SAM1-style interactive segmentation).
        See: https://samgeo.gishub.org/examples/sam3_box_prompts/

        Args:
            image_path: Path to input image (GeoTIFF supported)
            point_coords: Point prompts [[x, y], ...]
            point_labels: Point labels [1=foreground, 0=background]
            boxes: Box prompts [[x1, y1, x2, y2], ...] or single box [x1, y1, x2, y2]
            multimask_output: Whether to output multiple masks per prompt
            point_crs: CRS for point coordinates (e.g., "EPSG:4326")
            box_crs: CRS for box coordinates (e.g., "EPSG:4326")
            **kwargs: Additional parameters

        Returns:
            Dictionary containing masks, scores, and logits
        """
        if not self._loaded:
            self.load()

        try:
            logger.info(f"Running SAM prediction on {image_path}")

            # Set the image on the model
            self._set_image_if_needed(image_path)

            # Use predict_inst for both point and box prompts
            # This is the SAM1-style interactive segmentation API
            if point_coords is not None or boxes is not None:
                # Prepare point coordinates
                np_point_coords = None
                np_point_labels = None
                if point_coords is not None:
                    logger.info(f"Using point prompts: {point_coords}")
                    np_point_coords = np.array(point_coords)
                    np_point_labels = (
                        np.array(point_labels)
                        if point_labels is not None
                        else np.ones(len(point_coords), dtype=np.int32)
                    )

                # Prepare box coordinates
                np_box = None
                if boxes is not None:
                    logger.info(f"Using box prompts: {boxes}")
                    np_box = np.array(boxes)

                # Call predict_inst which returns (masks, scores, logits) tuple
                masks, scores, logits = self._model.predict_inst(
                    point_coords=np_point_coords,
                    point_labels=np_point_labels,
                    box=np_box,
                    multimask_output=multimask_output,
                    point_crs=point_crs,
                    box_crs=box_crs,
                    **kwargs,
                )

                return {
                    "masks": masks,
                    "scores": scores,
                    "logits": logits,
                }
            else:
                raise ValueError("Either point_coords or boxes must be provided")

        except Exception as e:
            logger.error(f"SAM prediction failed: {e}")
            raise

    def generate_masks(
        self,
        image_path: Path | str,
        prompt: str = "object",
        output_path: Path | str | None = None,
        min_size: int = 0,
        max_size: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run text-prompted mask generation using SAM3.

        SAM3 uses text prompts to segment objects, unlike SAM2 which uses
        grid-based point sampling.
        See: https://samgeo.gishub.org/examples/sam3_box_prompts/

        Args:
            image_path: Path to input image
            prompt: Text prompt describing what to segment (e.g., "tree", "building")
            output_path: Path to save output masks
            min_size: Minimum mask size in pixels
            max_size: Maximum mask size in pixels
            **kwargs: Additional parameters

        Returns:
            Dictionary containing generated masks and metadata
        """
        if not self._loaded:
            self.load()

        try:
            logger.info(
                f"Running SAM3 text-prompted mask generation on {image_path} "
                f"with prompt='{prompt}'"
            )

            # Set the image on the model
            self._set_image_if_needed(image_path)

            # Run text-prompted mask generation (SAM3 API)
            masks = self._model.generate_masks(
                prompt=prompt,
                min_size=min_size,
                max_size=max_size,
                **kwargs,
            )

            # Save masks if output path provided
            if output_path:
                self._model.save_masks(output=str(output_path))

            return {
                "status": "success",
                "masks": masks,
                "output_path": str(output_path) if output_path else None,
                "params": {
                    "prompt": prompt,
                    "min_size": min_size,
                    "max_size": max_size,
                },
            }

        except Exception as e:
            logger.error(f"SAM mask generation failed: {e}")
            raise

    def generate_masks_by_boxes(
        self,
        image_path: Path | str,
        boxes: list[list[float]] | str,
        box_crs: str | None = None,
        output_path: Path | str | None = None,
        multimask_output: bool = False,
        min_size: int = 0,
        max_size: int | None = None,
        dtype: str = "uint8",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate masks using bounding box prompts.

        This is the recommended method for batch box-based segmentation.
        See: https://samgeo.gishub.org/examples/sam3_box_prompts/

        Args:
            image_path: Path to input image (GeoTIFF supported)
            boxes: Box prompts as [[xmin, ymin, xmax, ymax], ...] or path to
                   vector file (GeoJSON, Shapefile, etc.)
            box_crs: CRS for box coordinates (e.g., "EPSG:4326")
            output_path: Path to save output masks
            multimask_output: Whether to output multiple masks per box
            min_size: Minimum mask size in pixels
            max_size: Maximum mask size in pixels
            dtype: Output data type for saved masks
            **kwargs: Additional parameters

        Returns:
            Dictionary containing masks, scores, and metadata
        """
        if not self._loaded:
            self.load()

        try:
            logger.info(f"Running SAM box-based mask generation on {image_path}")

            # Set the image on the model
            self._set_image_if_needed(image_path)

            # Call generate_masks_by_boxes_inst (SAM3 API for batch box prompts)
            result = self._model.generate_masks_by_boxes_inst(
                boxes=boxes,
                box_crs=box_crs,
                output=str(output_path) if output_path else None,
                multimask_output=multimask_output,
                min_size=min_size,
                max_size=max_size,
                dtype=dtype,
                **kwargs,
            )

            return {
                "status": "success",
                "masks": result.get("masks") if isinstance(result, dict) else result,
                "scores": result.get("scores") if isinstance(result, dict) else None,
                "output_path": str(output_path) if output_path else None,
                "params": {
                    "box_crs": box_crs,
                    "multimask_output": multimask_output,
                },
            }

        except Exception as e:
            logger.error(f"SAM box mask generation failed: {e}")
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
