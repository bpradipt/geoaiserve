"""Real model tests for DINOv3 (DINOv2) feature extraction service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoaiserve.utils.geotiff_loader import load_geotiff_as_rgb
from tests.markers import skip_without_ml

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.real_dinov3,
    pytest.mark.slow,
]


@skip_without_ml
class TestRealDINOv3Service:
    """Tests for DINOv3 service with real model inference."""

    def test_model_loads_successfully(self, real_dinov3_service):
        """DINOv3 model should load without errors."""
        assert real_dinov3_service is not None
        assert real_dinov3_service._loaded is True

    def test_model_type_is_dinov3(self, real_dinov3_service):
        """Model type should be DINOV3."""
        from geoaiserve.schemas.common import ModelType

        assert real_dinov3_service.model_type == ModelType.DINOV3

    def test_supported_tasks(self, real_dinov3_service):
        """Should support expected tasks."""
        tasks = real_dinov3_service.supported_tasks
        assert "feature_extraction" in tasks
        assert "patch_similarity" in tasks

    def test_extract_features_pil_image(self, real_dinov3_service):
        """Should extract features from PIL Image."""
        img = Image.new("RGB", (224, 224), color="blue")

        result = real_dinov3_service.extract_features(img)

        assert "cls_token" in result
        assert "feature_dim" in result
        assert isinstance(result["cls_token"], list)
        assert result["feature_dim"] > 0

    def test_feature_dimension(self, real_dinov3_service):
        """Features should have correct dimension for model."""
        img = Image.new("RGB", (224, 224), color="red")

        result = real_dinov3_service.extract_features(img)

        # DINOv2-base has 768-dim features
        assert result["feature_dim"] in [384, 768, 1024, 1536]

    def test_extract_features_with_patches(self, real_dinov3_service):
        """Should return patch features when requested."""
        img = Image.new("RGB", (224, 224), color="green")

        result = real_dinov3_service.extract_features(
            img, return_patch_features=True
        )

        assert "cls_token" in result
        assert "patch_features" in result

    def test_patch_grid_dimensions(self, real_dinov3_service):
        """Should return correct patch grid dimensions."""
        img = Image.new("RGB", (224, 224), color="cyan")

        result = real_dinov3_service.extract_features(img)

        assert "patch_grid" in result
        assert result["patch_grid"] is not None
        assert len(result["patch_grid"]) == 2
        # DINOv2 uses 14x14 patches, 224/14 = 16 patches per side
        assert result["patch_grid"][0] == 16
        assert result["patch_grid"][1] == 16

    def test_compute_similarity(self, real_dinov3_service):
        """Should compute similarity between two images."""
        img1 = Image.new("RGB", (224, 224), color="red")
        img2 = Image.new("RGB", (224, 224), color="blue")

        result = real_dinov3_service.compute_similarity(img1, img2)

        assert "similarity" in result
        assert isinstance(result["similarity"], float)
        assert -1.0 <= result["similarity"] <= 1.0

    def test_identical_images_high_similarity(self, real_dinov3_service):
        """Identical images should have high similarity."""
        img = Image.new("RGB", (224, 224), color="purple")

        result = real_dinov3_service.compute_similarity(img, img)

        # Identical images should have similarity close to 1.0
        assert result["similarity"] > 0.99


@skip_without_ml
class TestDINOv3FeatureConsistency:
    """Tests for DINOv3 feature consistency and determinism."""

    def test_features_are_deterministic(self, real_dinov3_service):
        """Same image should produce same features."""
        img = Image.new("RGB", (224, 224), color="orange")

        result1 = real_dinov3_service.extract_features(img)
        result2 = real_dinov3_service.extract_features(img)

        # Features should be identical for same input
        np.testing.assert_array_almost_equal(
            np.array(result1["cls_token"]),
            np.array(result2["cls_token"]),
            decimal=5,
        )

    def test_different_images_different_features(self, real_dinov3_service):
        """Different images should produce different features."""
        img1 = Image.new("RGB", (224, 224), color="white")
        img2 = Image.new("RGB", (224, 224), color="black")

        result1 = real_dinov3_service.extract_features(img1)
        result2 = real_dinov3_service.extract_features(img2)

        # Features should be different
        feat1 = np.array(result1["cls_token"])
        feat2 = np.array(result2["cls_token"])
        assert not np.allclose(feat1, feat2)

    def test_batch_similarity(self, real_dinov3_service):
        """Should compute batch similarity correctly."""
        query = Image.new("RGB", (224, 224), color="red")
        candidates = [
            Image.new("RGB", (224, 224), color="red"),   # Most similar
            Image.new("RGB", (224, 224), color="blue"),
            Image.new("RGB", (224, 224), color="green"),
        ]

        result = real_dinov3_service.batch_similarity(query, candidates)

        assert "num_candidates" in result
        assert result["num_candidates"] == 3
        assert "similarities" in result
        assert len(result["similarities"]) == 3

        # First candidate (red) should be most similar to query (red)
        # Results are sorted by similarity (descending)
        top_match = result["similarities"][0]
        assert top_match["index"] == 0  # Red image should be first


@skip_without_ml
class TestDINOv3PatchSimilarity:
    """Tests for DINOv3 patch similarity computation."""

    def test_query_patch_has_max_similarity(self, real_dinov3_service):
        """Query patch should have similarity 1.0 with itself."""
        img = Image.new("RGB", (224, 224), color="blue")

        # Query center of image (112, 112)
        result = real_dinov3_service.compute_patch_similarity(
            img, query_points=[[112, 112]]
        )

        assert "similarity_maps" in result
        assert len(result["similarity_maps"]) == 1

        sim_map = np.array(result["similarity_maps"][0])
        # The center patch should have max similarity (1.0)
        # Center of 16x16 grid is around (8, 8)
        center_x, center_y = 8, 8
        max_sim = sim_map.max()
        center_sim = sim_map[center_y, center_x]

        # Center should have the highest or near-highest similarity
        assert center_sim > 0.99, f"Center similarity should be ~1.0, got {center_sim}"

    def test_similarity_map_dimensions(self, real_dinov3_service):
        """Similarity maps should match patch grid dimensions."""
        img = Image.new("RGB", (224, 224), color="green")

        result = real_dinov3_service.compute_patch_similarity(
            img, query_points=[[50, 50]]
        )

        assert "map_size" in result
        assert result["map_size"] == [16, 16]

        sim_map = np.array(result["similarity_maps"][0])
        assert sim_map.shape == (16, 16)

    def test_multiple_query_points(self, real_dinov3_service):
        """Should handle multiple query points correctly."""
        img = Image.new("RGB", (224, 224), color="red")

        result = real_dinov3_service.compute_patch_similarity(
            img, query_points=[[50, 50], [150, 150], [100, 200]]
        )

        assert len(result["similarity_maps"]) == 3
        assert len(result["query_points"]) == 3

        for sim_map in result["similarity_maps"]:
            sim_array = np.array(sim_map)
            assert sim_array.shape == (16, 16)
            # Similarity values should be between -1 and 1
            assert sim_array.min() >= -1.0
            assert sim_array.max() <= 1.0


@skip_without_ml
@pytest.mark.geotiff
class TestDINOv3WithGeoTIFF:
    """Tests for DINOv3 with GeoTIFF satellite imagery."""

    def test_extract_features_from_geotiff(
        self, real_dinov3_service, sample_geotiff: Path | None
    ):
        """Should extract features from GeoTIFF files."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        geotiff_data = load_geotiff_as_rgb(sample_geotiff)

        result = real_dinov3_service.extract_features(geotiff_data.image)

        assert "cls_token" in result
        assert "feature_dim" in result

    def test_geotiff_similarity(
        self, real_dinov3_service, geotiff_paths: list[Path]
    ):
        """Should compute similarity between GeoTIFF files."""
        if len(geotiff_paths) < 2:
            pytest.skip("Need at least 2 GeoTIFF files for similarity test")

        img1 = load_geotiff_as_rgb(geotiff_paths[0]).image
        img2 = load_geotiff_as_rgb(geotiff_paths[1]).image

        result = real_dinov3_service.compute_similarity(img1, img2)

        assert "similarity" in result
        assert isinstance(result["similarity"], float)

    def test_patch_similarity_with_geotiff(
        self, real_dinov3_service, sample_geotiff: Path | None
    ):
        """Should compute patch similarity on GeoTIFF."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        geotiff_data = load_geotiff_as_rgb(sample_geotiff)

        result = real_dinov3_service.compute_patch_similarity(
            geotiff_data.image,
            query_points=[[100, 100], [200, 200]],
        )

        assert "query_points" in result
        assert "similarity_maps" in result
        assert len(result["similarity_maps"]) == 2
