"""Tests for Moondream vision-language model endpoints."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient


def test_moondream_caption(client: TestClient, sample_image: BytesIO) -> None:
    """Test Moondream image captioning endpoint.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/moondream/caption",
        files={"file": ("test.png", sample_image, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "caption" in data
    assert "length" in data
    assert "metadata" in data
    assert isinstance(data["caption"], str)
    assert len(data["caption"]) > 0


@pytest.mark.parametrize(
    "length",
    ["short", "normal", "long"],
)
def test_moondream_caption_lengths(
    client: TestClient,
    sample_image: BytesIO,
    length: str,
) -> None:
    """Test Moondream captioning with different lengths.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
        length: Caption length parameter
    """
    sample_image.seek(0)

    response = client.post(
        "/api/v1/moondream/caption",
        files={"file": ("test.png", sample_image, "image/png")},
        data={"length": length},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["length"] == length


def test_moondream_query(client: TestClient, sample_image: BytesIO) -> None:
    """Test Moondream visual question answering.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/moondream/query",
        files={"file": ("test.png", sample_image, "image/png")},
        json={"question": "What color is the image?"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["question"] == "What color is the image?"
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert "metadata" in data


@pytest.mark.parametrize(
    "question",
    [
        "What is in this image?",
        "How many objects are visible?",
        "What color is dominant?",
    ],
)
def test_moondream_query_variations(
    client: TestClient,
    sample_image: BytesIO,
    question: str,
) -> None:
    """Test Moondream VQA with different questions.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
        question: Question to ask
    """
    sample_image.seek(0)

    response = client.post(
        "/api/v1/moondream/query",
        files={"file": ("test.png", sample_image, "image/png")},
        json={"question": question},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == question


def test_moondream_detect(client: TestClient, sample_image: BytesIO) -> None:
    """Test Moondream object detection endpoint.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/moondream/detect",
        files={"file": ("test.png", sample_image, "image/png")},
        json={"object_type": "car"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["object_type"] == "car"
    assert "detections" in data
    assert "num_detections" in data
    assert isinstance(data["detections"], list)


def test_moondream_point(client: TestClient, sample_image: BytesIO) -> None:
    """Test Moondream point detection endpoint.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/moondream/point",
        files={"file": ("test.png", sample_image, "image/png")},
        json={"object_description": "the center"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["object_description"] == "the center"
    assert "points" in data
    assert "num_points" in data
    assert isinstance(data["points"], list)
