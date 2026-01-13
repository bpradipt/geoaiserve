"""Unit tests for GeoTIFF loading utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoaiserve.utils.geotiff_loader import (
    GeoTIFFData,
    _normalize_to_uint8,
    load_geotiff_as_rgb,
    load_geotiff_metadata,
    save_rgb_as_geotiff,
)


class TestNormalizeToUint8:
    """Tests for _normalize_to_uint8 helper function."""

    def test_uint8_passthrough(self):
        """uint8 arrays should pass through unchanged."""
        arr = np.array([[0, 128, 255]], dtype=np.uint8)
        result = _normalize_to_uint8(arr)
        np.testing.assert_array_equal(result, arr)

    def test_float_0_to_1_range(self):
        """Floats in 0-1 range should scale to 0-255."""
        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = _normalize_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_float_larger_range(self):
        """Floats outside 0-1 should be normalized first."""
        arr = np.array([[0.0, 500.0, 1000.0]], dtype=np.float32)
        result = _normalize_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_uint16_normalization(self):
        """uint16 arrays should normalize to 0-255."""
        arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
        result = _normalize_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_constant_array(self):
        """Constant arrays should return zeros."""
        arr = np.array([[100, 100, 100]], dtype=np.float32)
        result = _normalize_to_uint8(arr)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros_like(result))


@pytest.mark.geotiff
class TestGeoTIFFLoader:
    """Tests for load_geotiff_as_rgb function."""

    def test_file_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_geotiff_as_rgb(tmp_path / "nonexistent.tif")

    def test_load_returns_geotiff_data(self, sample_geotiff: Path | None):
        """Should return GeoTIFFData object with correct structure."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        result = load_geotiff_as_rgb(sample_geotiff)

        assert isinstance(result, GeoTIFFData)
        assert isinstance(result.image, Image.Image)
        assert result.image.mode == "RGB"
        assert isinstance(result.metadata, dict)
        assert isinstance(result.original_shape, tuple)
        assert isinstance(result.dtype, str)

    def test_metadata_contains_required_fields(self, sample_geotiff: Path | None):
        """Metadata should contain expected fields."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        result = load_geotiff_as_rgb(sample_geotiff)

        assert "width" in result.metadata
        assert "height" in result.metadata
        assert "count" in result.metadata
        assert "dtype" in result.metadata
        assert "crs" in result.metadata
        assert "bounds" in result.metadata
        assert "transform" in result.metadata

    def test_image_dimensions_match_metadata(self, sample_geotiff: Path | None):
        """Image dimensions should match metadata."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        result = load_geotiff_as_rgb(sample_geotiff)

        assert result.image.width == result.metadata["width"]
        assert result.image.height == result.metadata["height"]

    def test_invalid_band_raises_error(self, sample_geotiff: Path | None):
        """Should raise ValueError for out-of-range bands."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        with pytest.raises(ValueError, match="Band .* out of range"):
            load_geotiff_as_rgb(sample_geotiff, bands=(1, 2, 100))


@pytest.mark.geotiff
class TestGeoTIFFMetadata:
    """Tests for load_geotiff_metadata function."""

    def test_file_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_geotiff_metadata(tmp_path / "nonexistent.tif")

    def test_returns_metadata_dict(self, sample_geotiff: Path | None):
        """Should return dictionary with metadata."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        result = load_geotiff_metadata(sample_geotiff)

        assert isinstance(result, dict)
        assert "width" in result
        assert "height" in result
        assert "count" in result


@pytest.mark.geotiff
class TestGeoTIFFSaveRoundtrip:
    """Tests for save_rgb_as_geotiff function."""

    def test_save_creates_file(self, tmp_path: Path):
        """Should create GeoTIFF file on disk."""
        img = Image.new("RGB", (100, 100), color="red")
        output_path = tmp_path / "output.tif"

        result = save_rgb_as_geotiff(img, output_path)

        assert result.exists()
        assert result == output_path

    def test_save_with_metadata(self, tmp_path: Path, sample_geotiff: Path | None):
        """Should preserve georeferencing when metadata provided."""
        if sample_geotiff is None:
            pytest.skip("No GeoTIFF test files available")

        # Load original
        original = load_geotiff_as_rgb(sample_geotiff)

        # Save with metadata
        output_path = tmp_path / "output.tif"
        save_rgb_as_geotiff(original.image, output_path, original.metadata)

        # Verify file created
        assert output_path.exists()

        # Load saved file and check metadata
        saved_metadata = load_geotiff_metadata(output_path)
        assert saved_metadata["width"] == original.metadata["width"]
        assert saved_metadata["height"] == original.metadata["height"]

    def test_roundtrip_preserves_dimensions(self, tmp_path: Path):
        """Image dimensions should be preserved through save/load cycle."""
        # Create test image
        img = Image.new("RGB", (256, 128), color="blue")
        output_path = tmp_path / "roundtrip.tif"

        # Save
        save_rgb_as_geotiff(img, output_path)

        # Load back
        result = load_geotiff_as_rgb(output_path)

        assert result.image.width == 256
        assert result.image.height == 128
        assert result.image.mode == "RGB"

    def test_creates_parent_directories(self, tmp_path: Path):
        """Should create parent directories if they don't exist."""
        img = Image.new("RGB", (50, 50), color="green")
        output_path = tmp_path / "nested" / "dir" / "output.tif"

        result = save_rgb_as_geotiff(img, output_path)

        assert result.exists()
        assert result.parent.exists()
