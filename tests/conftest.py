"""Pytest configuration and fixtures."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geoaiserve.main import app
from tests.markers import ML_AVAILABLE, SAMGEO_AVAILABLE


@pytest.fixture
def client():
    """Create FastAPI test client with lifespan events.

    Yields:
        TestClient instance for making test requests
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_image() -> BytesIO:
    """Create a sample test image.

    Returns:
        BytesIO containing a test image
    """
    # Create a simple RGB image
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_image_file(tmp_path: Path) -> Path:
    """Create a sample image file on disk.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to the created test image
    """
    img = Image.new("RGB", (100, 100), color="blue")
    img_path = tmp_path / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def uploaded_file_id(client: TestClient, sample_image: BytesIO) -> str:
    """Upload a sample image and return the file_id.

    Args:
        client: FastAPI test client
        sample_image: Sample test image

    Returns:
        file_id string for the uploaded image
    """
    sample_image.seek(0)
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    return response.json()["file_id"]


@pytest.fixture
def uploaded_file_ids(client: TestClient, sample_image: BytesIO) -> list[str]:
    """Upload multiple sample images and return file_ids.

    Args:
        client: FastAPI test client
        sample_image: Sample test image

    Returns:
        List of file_id strings for uploaded images
    """
    file_ids = []
    for i in range(3):
        sample_image.seek(0)
        response = client.post(
            "/api/v1/files/upload",
            files={"file": (f"test_{i}.png", sample_image, "image/png")},
        )
        file_ids.append(response.json()["file_id"])
    return file_ids


@pytest.fixture
def invalid_file_id() -> str:
    """Return a non-existent file ID for error testing.

    Returns:
        A fake UUID that doesn't exist in the system
    """
    return "nonexistent-uuid-12345-abcdef"


@pytest.fixture
def sample_image_small() -> BytesIO:
    """Create a minimal 1x1 pixel image for edge case testing.

    Returns:
        BytesIO containing a 1x1 pixel PNG image
    """
    img = Image.new("RGB", (1, 1), color="white")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_image_large() -> BytesIO:
    """Create a larger image for testing.

    Returns:
        BytesIO containing a 500x500 pixel PNG image
    """
    img = Image.new("RGB", (500, 500), color="green")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def corrupted_image_bytes() -> BytesIO:
    """Create corrupted/invalid image bytes for error testing.

    Returns:
        BytesIO containing invalid image data
    """
    return BytesIO(b"not a valid image content at all")


# ============================================================================
# GeoTIFF and Real Model Testing Fixtures
# ============================================================================


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory.

    Returns:
        Path to tests/data/ directory
    """
    return Path(__file__).parent / "data"


@pytest.fixture
def geotiff_paths(test_data_dir: Path) -> list[Path]:
    """List of available GeoTIFF files in test data directory.

    Args:
        test_data_dir: Path to test data directory

    Returns:
        List of paths to .tif/.tiff files
    """
    tif_files = list(test_data_dir.glob("*.tif"))
    tiff_files = list(test_data_dir.glob("*.tiff"))
    return sorted(tif_files + tiff_files)


@pytest.fixture
def sample_geotiff(geotiff_paths: list[Path]) -> Path | None:
    """First available GeoTIFF file for testing.

    Args:
        geotiff_paths: List of GeoTIFF paths

    Returns:
        Path to first GeoTIFF or None if none available
    """
    if geotiff_paths:
        return geotiff_paths[0]
    return None


@pytest.fixture
def uploaded_geotiff_id(
    client: TestClient, sample_geotiff: Path | None
) -> str | None:
    """Upload a GeoTIFF file and return the file_id.

    Args:
        client: FastAPI test client
        sample_geotiff: Path to sample GeoTIFF

    Returns:
        file_id string or None if no GeoTIFF available
    """
    if sample_geotiff is None:
        return None

    with open(sample_geotiff, "rb") as f:
        response = client.post(
            "/api/v1/files/upload",
            files={"file": (sample_geotiff.name, f, "image/tiff")},
        )
    return response.json()["file_id"]


@pytest.fixture(scope="session")
def real_sam_service():
    """Real SAM model service (session-scoped for efficiency).

    Returns:
        SAMService instance with real model or None if not available
    """
    if not SAMGEO_AVAILABLE:
        return None

    from geoaiserve.models.sam_service import SAMService
    from geoaiserve.schemas.common import DeviceType

    service = SAMService(device=DeviceType.CPU)
    service.load()
    yield service
    service.unload()


@pytest.fixture(scope="session")
def real_moondream_service():
    """Real Moondream model service (session-scoped for efficiency).

    Returns:
        MoondreamService instance with real model or None if not available
    """
    if not ML_AVAILABLE:
        return None

    from geoaiserve.models.moondream_service import MoondreamService
    from geoaiserve.schemas.common import DeviceType

    service = MoondreamService(device=DeviceType.CPU)
    service.load()
    yield service
    service.unload()


@pytest.fixture(scope="session")
def real_dinov3_service():
    """Real DINOv3 model service (session-scoped for efficiency).

    Returns:
        DINOv3Service instance with real model or None if not available
    """
    if not ML_AVAILABLE:
        return None

    from geoaiserve.models.dinov3_service import DINOv3Service
    from geoaiserve.schemas.common import DeviceType

    service = DINOv3Service(device=DeviceType.CPU)
    service.load()
    yield service
    service.unload()
