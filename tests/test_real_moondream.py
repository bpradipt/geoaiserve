"""Real model tests for Moondream vision-language service."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from geoaiserve.utils.geotiff_loader import load_geotiff_as_rgb
from tests.markers import skip_without_ml

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.real_moondream,
    pytest.mark.slow,
]


@skip_without_ml
class TestRealMoondreamService:
    """Tests for Moondream service with real model inference."""

    def test_model_loads_successfully(self, real_moondream_service):
        """Moondream model should load without errors."""
        assert real_moondream_service is not None
        assert real_moondream_service._loaded is True

    def test_model_type_is_moondream(self, real_moondream_service):
        """Model type should be MOONDREAM."""
        from geoaiserve.schemas.common import ModelType

        assert real_moondream_service.model_type == ModelType.MOONDREAM

    def test_supported_tasks(self, real_moondream_service):
        """Should support expected tasks."""
        tasks = real_moondream_service.supported_tasks
        assert "image_captioning" in tasks
        assert "visual_qa" in tasks
        assert "object_detection" in tasks

    def test_caption_with_pil_image(self, real_moondream_service):
        """Should generate caption from PIL Image."""
        # Create a simple test image
        img = Image.new("RGB", (224, 224), color="blue")

        result = real_moondream_service.caption(img)

        assert "caption" in result
        assert isinstance(result["caption"], str)
        assert len(result["caption"]) > 0

    def test_caption_with_geotiff(
        self, real_moondream_service, sample_geotiff: Path | None
    ):
        """Should generate caption from GeoTIFF file."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        # Load GeoTIFF as RGB
        geotiff_data = load_geotiff_as_rgb(sample_geotiff)

        result = real_moondream_service.caption(geotiff_data.image)

        assert "caption" in result
        assert isinstance(result["caption"], str)

    def test_query_answers_question(self, real_moondream_service):
        """Should answer visual questions about images."""
        img = Image.new("RGB", (224, 224), color="green")

        result = real_moondream_service.query(img, "What color is this image?")

        assert "question" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_detect_returns_objects(self, real_moondream_service):
        """Should detect objects in images."""
        img = Image.new("RGB", (224, 224), color="red")

        result = real_moondream_service.detect(img, "shapes")

        assert "object_type" in result
        assert "detections" in result

    def test_caption_length_options(self, real_moondream_service):
        """Should respect caption length parameter."""
        img = Image.new("RGB", (224, 224), color="yellow")

        short_result = real_moondream_service.caption(img, length="short")
        normal_result = real_moondream_service.caption(img, length="normal")

        assert "caption" in short_result
        assert "caption" in normal_result
        assert short_result["length"] == "short"
        assert normal_result["length"] == "normal"


@skip_without_ml
@pytest.mark.geotiff
class TestMoondreamWithGeoTIFF:
    """Tests for Moondream with GeoTIFF satellite imagery."""

    def test_satellite_image_caption(
        self, real_moondream_service, sample_geotiff: Path | None
    ):
        """Should generate meaningful caption for satellite imagery."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        geotiff_data = load_geotiff_as_rgb(sample_geotiff)

        result = real_moondream_service.caption(geotiff_data.image)

        assert "caption" in result
        # Caption should be non-empty
        assert len(result["caption"].strip()) > 0

    def test_satellite_qa(
        self, real_moondream_service, sample_geotiff: Path | None
    ):
        """Should answer questions about satellite imagery."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        geotiff_data = load_geotiff_as_rgb(sample_geotiff)

        result = real_moondream_service.query(
            geotiff_data.image,
            "What type of terrain or features are visible in this image?",
        )

        assert "answer" in result
        assert len(result["answer"].strip()) > 0
