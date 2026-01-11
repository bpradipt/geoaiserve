"""Tests for Moondream vision-language model endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_moondream_caption(client: TestClient, uploaded_file_id: str) -> None:
    """Test Moondream image captioning endpoint.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/caption",
        json={"file_id": uploaded_file_id},
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
    uploaded_file_id: str,
    length: str,
) -> None:
    """Test Moondream captioning with different lengths.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        length: Caption length parameter
    """
    response = client.post(
        "/api/v1/moondream/caption",
        json={
            "file_id": uploaded_file_id,
            "length": length,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["length"] == length


def test_moondream_query(client: TestClient, uploaded_file_id: str) -> None:
    """Test Moondream visual question answering.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/query",
        json={
            "file_id": uploaded_file_id,
            "question": "What color is the image?",
        },
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
    uploaded_file_id: str,
    question: str,
) -> None:
    """Test Moondream VQA with different questions.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        question: Question to ask
    """
    response = client.post(
        "/api/v1/moondream/query",
        json={
            "file_id": uploaded_file_id,
            "question": question,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == question


def test_moondream_detect(client: TestClient, uploaded_file_id: str) -> None:
    """Test Moondream object detection endpoint.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/detect",
        json={
            "file_id": uploaded_file_id,
            "object_type": "car",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["object_type"] == "car"
    assert "detections" in data
    assert "num_detections" in data
    assert isinstance(data["detections"], list)


def test_moondream_point(client: TestClient, uploaded_file_id: str) -> None:
    """Test Moondream point detection endpoint.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/point",
        json={
            "file_id": uploaded_file_id,
            "object_description": "the center",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["object_description"] == "the center"
    assert "points" in data
    assert "num_points" in data
    assert isinstance(data["points"], list)


def test_moondream_caption_without_file_id(client: TestClient) -> None:
    """Test Moondream caption fails without file_id.

    Args:
        client: FastAPI test client
    """
    response = client.post(
        "/api/v1/moondream/caption",
        json={"length": "normal"},
    )

    # Should return 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
