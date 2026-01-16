"""Real model tests for Moondream vision-language service.

These tests verify actual Moondream model inference using satellite imagery.
Tests use parking_lot.tif which contains parking areas, buildings, and vegetation.
"""

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


# =============================================================================
# Fixtures for Moondream Tests
# =============================================================================


@pytest.fixture
def parking_lot_image(test_data_dir: Path) -> Image.Image | None:
    """Load parking_lot.tif as PIL Image for testing.

    Returns:
        PIL Image or None if file not available
    """
    parking_lot_path = test_data_dir / "parking_lot.tif"
    if not parking_lot_path.exists():
        return None
    geotiff_data = load_geotiff_as_rgb(parking_lot_path)
    return geotiff_data.image


@pytest.fixture
def parking_lot_path(test_data_dir: Path) -> Path | None:
    """Path to parking_lot.tif for testing.

    Returns:
        Path or None if file not available
    """
    parking_lot_path = test_data_dir / "parking_lot.tif"
    if not parking_lot_path.exists():
        return None
    return parking_lot_path


# =============================================================================
# Model Loading Tests
# =============================================================================


@skip_without_ml
class TestMoondreamModelLoading:
    """Tests for Moondream model initialization and loading."""

    def test_model_loads_successfully(self, real_moondream_service):
        """Moondream model should load without errors."""
        assert real_moondream_service is not None
        assert real_moondream_service._loaded is True

    def test_model_is_not_mock(self, real_moondream_service):
        """Real model fixture should not use mock."""
        assert real_moondream_service.is_mock is False

    def test_model_type_is_moondream(self, real_moondream_service):
        """Model type should be MOONDREAM."""
        from geoaiserve.schemas.common import ModelType

        assert real_moondream_service.model_type == ModelType.MOONDREAM

    def test_supported_tasks(self, real_moondream_service):
        """Should support expected vision-language tasks."""
        tasks = real_moondream_service.supported_tasks
        assert "image_captioning" in tasks
        assert "visual_qa" in tasks
        assert "object_detection" in tasks
        assert "point_detection" in tasks


# =============================================================================
# Image Captioning Tests
# =============================================================================


@skip_without_ml
@pytest.mark.geotiff
class TestMoondreamCaption:
    """Tests for Moondream image captioning with satellite imagery."""

    def test_caption_returns_valid_response(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Caption should return properly structured response."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.caption(parking_lot_image)

        assert "caption" in result
        assert "length" in result
        assert isinstance(result["caption"], str)
        assert len(result["caption"]) > 10  # Meaningful caption

    def test_caption_describes_parking_lot(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Caption should mention relevant features of parking lot imagery."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.caption(parking_lot_image, length="normal")

        caption = result["caption"].lower()
        # Caption should mention at least one relevant feature
        relevant_terms = [
            "parking",
            "car",
            "vehicle",
            "building",
            "lot",
            "aerial",
            "road",
            "asphalt",
            "structure",
        ]
        found_terms = [term for term in relevant_terms if term in caption]
        assert (
            len(found_terms) > 0
        ), f"Caption '{result['caption']}' should mention parking-related features"

    def test_caption_length_short(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Short captions should be concise."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.caption(parking_lot_image, length="short")

        assert result["length"] == "short"
        assert len(result["caption"]) > 0

    def test_caption_length_normal(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Normal captions should have moderate detail."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.caption(parking_lot_image, length="normal")

        assert result["length"] == "normal"
        assert len(result["caption"]) > 0


# =============================================================================
# Visual Question Answering Tests
# =============================================================================


@skip_without_ml
@pytest.mark.geotiff
class TestMoondreamQuery:
    """Tests for Moondream visual question answering with satellite imagery."""

    def test_query_returns_valid_response(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Query should return properly structured response."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.query(
            parking_lot_image, "What is in this image?"
        )

        assert "question" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_about_vehicles(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Model should answer questions about vehicles in parking lot."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.query(
            parking_lot_image, "Are there any cars or vehicles visible in this image?"
        )

        assert "answer" in result
        # Answer should acknowledge the question meaningfully
        assert len(result["answer"]) > 5

    def test_query_about_buildings(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Model should answer questions about buildings."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.query(
            parking_lot_image, "Are there any buildings in this image?"
        )

        assert "answer" in result
        assert len(result["answer"]) > 5

    def test_query_counting(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Model should attempt to count objects."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.query(
            parking_lot_image, "How many vehicles are visible in this image?"
        )

        assert "answer" in result
        # Answer should contain some response (might be a number or description)
        assert len(result["answer"]) > 0

    def test_query_about_colors(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Model should describe colors in the image."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.query(
            parking_lot_image, "What colors are the roofs of any buildings?"
        )

        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_query_about_land_use(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Model should describe land use and features."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.query(
            parking_lot_image,
            "Describe the land use and features visible in this aerial image.",
        )

        assert "answer" in result
        # Should provide a descriptive answer
        assert len(result["answer"]) > 20


# =============================================================================
# Object Detection Tests
# =============================================================================


@skip_without_ml
@pytest.mark.geotiff
class TestMoondreamDetect:
    """Tests for Moondream object detection with satellite imagery.

    Note: The detect method may return empty results depending on the model
    revision and the image content. These tests verify the API structure
    rather than detection accuracy.
    """

    def test_detect_returns_valid_response(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Detection should return properly structured response."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.detect(parking_lot_image, "car")

        assert "object_type" in result
        assert "detections" in result
        assert result["object_type"] == "car"
        # Detections should be a list of bounding boxes
        assert isinstance(result["detections"], list)

    def test_detect_cars(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Should return detection structure for cars."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.detect(parking_lot_image, "car")

        assert "detections" in result
        assert result["detections"] is not None
        # Detections is a list of bounding box dicts
        assert isinstance(result["detections"], list)

    def test_detect_buildings(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Should return detection structure for buildings."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.detect(parking_lot_image, "building")

        assert "detections" in result
        assert result["object_type"] == "building"
        assert isinstance(result["detections"], list)

    def test_detect_vehicles(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Should return detection structure for vehicles."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.detect(parking_lot_image, "vehicle")

        assert "detections" in result
        assert result["object_type"] == "vehicle"
        assert isinstance(result["detections"], list)


# =============================================================================
# Point Detection Tests
# =============================================================================


@skip_without_ml
@pytest.mark.geotiff
class TestMoondreamPoint:
    """Tests for Moondream point detection with satellite imagery.

    Note: Point detection may not be available in all model revisions.
    The 2024-08-26 revision does not support point detection.
    These tests verify the API structure handles missing support gracefully.
    """

    def test_point_returns_valid_response(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Point detection should return properly structured response."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.point(parking_lot_image, "car")

        assert "object_description" in result
        assert "points" in result
        assert result["object_description"] == "car"
        # Points should be a list (possibly empty if not supported)
        assert isinstance(result["points"], list)

    def test_point_returns_list_structure(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Point detection should return a list (possibly empty)."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.point(parking_lot_image, "vehicle")

        assert "points" in result
        assert isinstance(result["points"], list)
        # If points are returned, each should be a coordinate pair
        if len(result["points"]) > 0:
            assert len(result["points"][0]) == 2  # [x, y]

    def test_point_handles_any_object(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Point detection should handle any object description."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.point(parking_lot_image, "tree")

        assert "points" in result
        assert isinstance(result["points"], list)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@skip_without_ml
class TestMoondreamEdgeCases:
    """Tests for edge cases and error handling."""

    def test_caption_with_small_image(self, real_moondream_service):
        """Should handle small images gracefully."""
        small_img = Image.new("RGB", (64, 64), color="gray")

        result = real_moondream_service.caption(small_img)

        assert "caption" in result
        assert isinstance(result["caption"], str)

    def test_query_with_empty_question(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Should handle empty or minimal questions."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.query(parking_lot_image, "Describe this.")

        assert "answer" in result

    def test_detect_nonexistent_object(
        self, real_moondream_service, parking_lot_image: Image.Image | None
    ):
        """Should handle detection of objects not in image."""
        if parking_lot_image is None:
            pytest.skip("parking_lot.tif not available")

        result = real_moondream_service.detect(parking_lot_image, "elephant")

        assert "detections" in result
        # Should return valid structure even for non-existent objects
        assert isinstance(result["detections"], list)
