"""Pytest configuration and fixtures."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geoaiserve.main import app


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
