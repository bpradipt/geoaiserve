"""DINOv3 (DINOv2) feature extraction service implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..schemas.common import DeviceType, ModelType
from .base import BaseGeoModel

logger = logging.getLogger(__name__)


class DINOv3Service(BaseGeoModel):
    """Service for DINOv3 feature extraction and similarity analysis."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: DeviceType = DeviceType.CPU,
        **kwargs: Any,
    ):
        """Initialize DINOv3 service.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, device, **kwargs)
        self.patch_size = kwargs.get("patch_size", 14)

    @property
    def model_type(self) -> ModelType:
        """Return the model type."""
        return ModelType.DINOV3

    @property
    def supported_tasks(self) -> list[str]:
        """Return list of supported tasks."""
        return [
            "feature_extraction",
            "patch_similarity",
            "batch_similarity",
        ]

    def load(self) -> None:
        """Load the DINOv3 model into memory."""
        if self._loaded:
            logger.info(f"DINOv3 model already loaded: {self.model_name}")
            return

        try:
            logger.info(f"Loading DINOv3 model: {self.model_name} on {self.device}")

            try:
                from transformers import AutoImageProcessor, AutoModel

                # Load model and processor
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)

                # Move to device
                if self.device == DeviceType.CUDA:
                    self._model = self._model.to("cuda")
                elif self.device == DeviceType.MPS:
                    self._model = self._model.to("mps")

                self._loaded = True
                logger.info(f"DINOv3 model loaded successfully: {self.model_name}")

            except ImportError as e:
                if self._allow_mock:
                    logger.warning(
                        "transformers not installed. Creating mock DINOv3 service. "
                        "Set allow_mock=False or unset GEOAI_ALLOW_MOCK to require real model."
                    )
                    self._model = self._create_mock_model()
                    self._processor = None
                    self._is_mock = True
                    self._loaded = True
                else:
                    raise ImportError(
                        f"Required dependency 'transformers' not installed for DINOv3. "
                        f"Install with: uv sync --group ml"
                    ) from e

        except Exception as e:
            logger.error(f"Failed to load DINOv3 model: {e}")
            raise

    def _create_mock_model(self) -> Any:
        """Create a mock model for testing."""

        class MockDINOv3:
            def extract_features(self, image):
                # Return mock features (768-dim for base model)
                return np.random.randn(1, 768).astype(np.float32)

            def compute_similarity(self, features1, features2):
                # Return mock similarity score
                return 0.85

        return MockDINOv3()

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._loaded:
            logger.info(f"Unloading DINOv3 model: {self.model_name}")
            self._model = None
            self._processor = None
            self._loaded = False

    def predict(
        self,
        image_path: Path | str,
        task: str = "feature_extraction",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run inference based on task type.

        Args:
            image_path: Path to input image
            task: Task type (features, similarity)
            **kwargs: Task-specific parameters

        Returns:
            Dictionary containing task results
        """
        if not self._loaded:
            self.load()

        if task == "feature_extraction":
            return self.extract_features(image_path, **kwargs)
        elif task == "patch_similarity":
            return self.compute_patch_similarity(image_path, **kwargs)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def extract_features(
        self,
        image: Image.Image | Path | str,
        return_patch_features: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract features from an image.

        Args:
            image: PIL Image or path to image
            return_patch_features: Whether to return patch-level features
            **kwargs: Additional parameters

        Returns:
            Dictionary containing features
        """
        if not self._loaded:
            self.load()

        try:
            if isinstance(image, (Path, str)):
                image = Image.open(image).convert("RGB")

            logger.info("Extracting DINOv3 features")

            if hasattr(self._model, "extract_features"):
                # Mock model
                features = self._model.extract_features(image)
                return {
                    "cls_token": features.flatten().tolist(),
                    "patch_features": None,
                    "feature_dim": features.shape[-1],
                    "patch_grid": None,
                }
            else:
                # Real model
                import torch

                # Preprocess image
                inputs = self._processor(images=image, return_tensors="pt")

                # Move to device
                if self.device == DeviceType.CUDA:
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                elif self.device == DeviceType.MPS:
                    inputs = {k: v.to("mps") for k, v in inputs.items()}

                # Extract features
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Get CLS token and patch features
                # Squeeze batch dimension for single images
                cls_token = outputs.last_hidden_state[:, 0].cpu().numpy()
                cls_token = cls_token.squeeze(0)  # [batch, dim] -> [dim]

                # Calculate patch grid dimensions
                # DINOv2 uses 14x14 patches, input resized to 224x224
                num_patches = outputs.last_hidden_state.shape[1] - 1  # Exclude CLS token
                grid_size = int(np.sqrt(num_patches))
                patch_grid = [grid_size, grid_size]

                patch_features = None
                if return_patch_features:
                    # [batch, num_patches, dim] -> [num_patches, dim]
                    patch_features = outputs.last_hidden_state[:, 1:].cpu().numpy()
                    patch_features = patch_features.squeeze(0)

                return {
                    "cls_token": cls_token.tolist(),
                    "patch_features": patch_features.tolist() if patch_features is not None else None,
                    "feature_dim": int(cls_token.shape[-1]),
                    "patch_grid": patch_grid,
                }

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    def compute_patch_similarity(
        self,
        image: Image.Image | Path | str,
        query_points: list[list[float]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute patch similarity for query points.

        Args:
            image: PIL Image or path to image
            query_points: List of query points [[x, y], ...]
            **kwargs: Additional parameters

        Returns:
            Dictionary containing similarity maps
        """
        if not self._loaded:
            self.load()

        try:
            if isinstance(image, (Path, str)):
                image = Image.open(image).convert("RGB")

            logger.info(f"Computing patch similarity for {len(query_points)} points")

            # Extract patch features
            features_result = self.extract_features(image, return_patch_features=True)

            # For mock model (patch_features is None)
            if features_result["patch_features"] is None:
                # Return mock similarity maps
                similarity_maps = [
                    np.random.rand(16, 16).tolist()  # Mock similarity map
                    for _ in query_points
                ]
                return {
                    "query_points": query_points,
                    "similarity_maps": similarity_maps,
                    "map_size": [16, 16],
                }

            # For real model - compute actual similarity
            patch_features = np.array(features_result["patch_features"])
            num_patches = patch_features.shape[0]
            feature_dim = patch_features.shape[1]

            # Calculate patch grid dimensions
            # DINOv2 uses 14x14 patches, input is resized to 224x224
            # So for 224x224 image: 224/14 = 16 patches per side
            grid_size = int(np.sqrt(num_patches))
            h_patches, w_patches = grid_size, grid_size

            # Normalize patch features for cosine similarity
            patch_norms = np.linalg.norm(patch_features, axis=1, keepdims=True) + 1e-8
            patch_features_norm = patch_features / patch_norms

            # Get original image size for coordinate mapping
            img_width, img_height = image.size

            # Compute similarity maps for each query point
            similarity_maps = []
            for point in query_points:
                x, y = point[0], point[1]

                # Map pixel coordinates to patch indices
                # The model input is resized to 224x224, then divided into patches
                patch_x = int((x / img_width) * w_patches)
                patch_y = int((y / img_height) * h_patches)

                # Clamp to valid range
                patch_x = max(0, min(patch_x, w_patches - 1))
                patch_y = max(0, min(patch_y, h_patches - 1))

                # Get query patch index (row-major order)
                query_patch_idx = patch_y * w_patches + patch_x

                # Get query patch features (already normalized)
                query_feat = patch_features_norm[query_patch_idx]

                # Compute cosine similarity with all patches
                similarities = np.dot(patch_features_norm, query_feat)

                # Reshape to 2D grid
                similarity_map = similarities.reshape(h_patches, w_patches)
                similarity_maps.append(similarity_map.tolist())

            return {
                "query_points": query_points,
                "similarity_maps": similarity_maps,
                "map_size": [h_patches, w_patches],
            }

        except Exception as e:
            logger.error(f"Patch similarity computation failed: {e}")
            raise

    def compute_similarity(
        self,
        image1: Image.Image | Path | str,
        image2: Image.Image | Path | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute similarity between two images.

        Args:
            image1: First image
            image2: Second image
            **kwargs: Additional parameters

        Returns:
            Dictionary containing similarity score
        """
        if not self._loaded:
            self.load()

        try:
            # Extract features from both images
            features1 = self.extract_features(image1)
            features2 = self.extract_features(image2)

            # Compute cosine similarity
            feat1 = np.array(features1["cls_token"])
            feat2 = np.array(features2["cls_token"])

            # Normalize and compute dot product
            feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
            feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)

            similarity = float(np.dot(feat1_norm.flatten(), feat2_norm.flatten()))

            logger.info(f"Computed similarity: {similarity:.4f}")

            return {
                "similarity": similarity,
                "image1_features": features1["cls_token"],
                "image2_features": features2["cls_token"],
            }

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise

    def batch_similarity(
        self,
        query_image: Image.Image | Path | str,
        candidate_images: list[Image.Image | Path | str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute similarity between query image and multiple candidates.

        Args:
            query_image: Query image
            candidate_images: List of candidate images
            **kwargs: Additional parameters

        Returns:
            Dictionary containing similarity scores
        """
        if not self._loaded:
            self.load()

        try:
            logger.info(f"Computing batch similarity for {len(candidate_images)} candidates")

            # Extract query features
            query_features = self.extract_features(query_image)
            query_feat = np.array(query_features["cls_token"])
            query_feat_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)

            # Compute similarities
            similarities = []
            for idx, candidate in enumerate(candidate_images):
                try:
                    cand_features = self.extract_features(candidate)
                    cand_feat = np.array(cand_features["cls_token"])
                    cand_feat_norm = cand_feat / (np.linalg.norm(cand_feat) + 1e-8)

                    similarity = float(np.dot(query_feat_norm.flatten(), cand_feat_norm.flatten()))
                    similarities.append({
                        "index": idx,
                        "similarity": similarity,
                    })
                except Exception as e:
                    logger.warning(f"Failed to process candidate {idx}: {e}")
                    similarities.append({
                        "index": idx,
                        "similarity": 0.0,
                        "error": str(e),
                    })

            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            return {
                "num_candidates": len(candidate_images),
                "similarities": similarities,
            }

        except Exception as e:
            logger.error(f"Batch similarity computation failed: {e}")
            raise
