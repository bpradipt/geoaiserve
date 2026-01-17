"""Real model tests for SAM (Segment Anything Model) service."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.markers import skip_without_samgeo

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.real_sam,
    pytest.mark.slow,
]


@skip_without_samgeo
class TestRealSAMService:
    """Tests for SAM service with real model inference."""

    def test_model_loads_successfully(self, real_sam_service):
        """SAM model should load without errors."""
        assert real_sam_service is not None
        assert real_sam_service._loaded is True

    def test_model_type_is_sam(self, real_sam_service):
        """Model type should be SAM."""
        from geoaiserve.schemas.common import ModelType

        assert real_sam_service.model_type == ModelType.SAM

    def test_supported_tasks(self, real_sam_service):
        """Should support expected tasks."""
        tasks = real_sam_service.supported_tasks
        assert "text_prompted_segmentation" in tasks
        assert "point_based_segmentation" in tasks
        assert "box_based_segmentation" in tasks

    def test_predict_with_geotiff(
        self, real_sam_service, sample_geotiff: Path | None
    ):
        """SAM should generate predictions from GeoTIFF input."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        # Run prediction with point prompt
        result = real_sam_service.predict(
            sample_geotiff,
            point_coords=[[100, 100]],
            point_labels=[1],
        )

        assert "masks" in result
        assert "scores" in result

    def test_predict_with_box_prompt(
        self, real_sam_service, sample_geotiff: Path | None
    ):
        """SAM should generate predictions from box prompts."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        result = real_sam_service.predict(
            sample_geotiff,
            boxes=[[50, 50, 150, 150]],
        )

        assert "masks" in result
        assert "scores" in result

    def test_generate_masks_with_text_prompt(
        self, real_sam_service, sample_geotiff: Path | None, tmp_path: Path
    ):
        """SAM3 should generate masks using text prompts."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        output_path = tmp_path / "masks_output.tif"

        # SAM3 uses text prompts for segmentation
        result = real_sam_service.generate_masks(
            sample_geotiff,
            prompt="object",  # Generic prompt for testing
            output_path=output_path,
        )

        assert result["status"] == "success"
        assert "params" in result
        assert result["params"]["prompt"] == "object"


@skip_without_samgeo
@pytest.mark.geotiff
class TestSAMWithMultipleGeoTIFFs:
    """Tests for SAM with multiple GeoTIFF files."""

    def test_batch_processing(
        self, real_sam_service, geotiff_paths: list[Path]
    ):
        """SAM should process multiple images in batch."""
        if not geotiff_paths:
            pytest.skip("No GeoTIFF test files available")

        # Process up to 2 files for speed
        test_paths = geotiff_paths[:2]

        results = real_sam_service.predict_batch(
            test_paths,
            point_coords=[[100, 100]],
            point_labels=[1],
        )

        assert len(results) == len(test_paths)
        for result in results:
            assert "image" in result
            assert "result" in result or "error" in result
