"""Tests for Moondream vision-language model endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.mock


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


# --- Parameter Validation Tests ---


def test_moondream_query_empty_question(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test Moondream query rejects empty question.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/query",
        json={
            "file_id": uploaded_file_id,
            "question": "",
        },
    )

    # Empty question should fail validation (min_length=1)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_moondream_detect_empty_object_type(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test Moondream detect rejects empty object_type.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/detect",
        json={
            "file_id": uploaded_file_id,
            "object_type": "",
        },
    )

    # Empty object_type should fail validation (min_length=1)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_moondream_point_empty_description(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test Moondream point rejects empty object_description.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/point",
        json={
            "file_id": uploaded_file_id,
            "object_description": "",
        },
    )

    # Empty description should fail validation (min_length=1)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.parametrize("confidence_threshold", [-0.1, 1.5, -1.0, 2.0])
def test_moondream_detect_invalid_confidence(
    client: TestClient,
    uploaded_file_id: str,
    confidence_threshold: float,
) -> None:
    """Test Moondream detect rejects invalid confidence_threshold.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        confidence_threshold: Invalid threshold (must be 0.0-1.0)
    """
    response = client.post(
        "/api/v1/moondream/detect",
        json={
            "file_id": uploaded_file_id,
            "object_type": "car",
            "confidence_threshold": confidence_threshold,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


# --- Non-existent file_id tests ---


def test_moondream_caption_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test Moondream caption returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/moondream/caption",
        json={"file_id": invalid_file_id},
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_moondream_query_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test Moondream query returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/moondream/query",
        json={
            "file_id": invalid_file_id,
            "question": "What is in this image?",
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_moondream_detect_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test Moondream detect returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/moondream/detect",
        json={
            "file_id": invalid_file_id,
            "object_type": "car",
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_moondream_point_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test Moondream point returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/moondream/point",
        json={
            "file_id": invalid_file_id,
            "object_description": "the center",
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


# --- Response Structure Validation ---


def test_moondream_caption_response_structure(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test Moondream caption response has correct structure.

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

    # Verify response structure
    assert data["status"] == "success"
    assert "caption" in data
    assert isinstance(data["caption"], str)
    assert len(data["caption"]) > 0
    assert "length" in data
    assert data["length"] in ["short", "normal", "long"]

    # Verify metadata
    assert "metadata" in data
    assert "processing_time" in data["metadata"]
    assert isinstance(data["metadata"]["processing_time"], (int, float))
    assert data["metadata"]["processing_time"] >= 0


def test_moondream_detect_response_structure(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test Moondream detect response has correct structure.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/detect",
        json={
            "file_id": uploaded_file_id,
            "object_type": "object",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert data["status"] == "success"
    assert data["object_type"] == "object"
    assert "detections" in data
    assert isinstance(data["detections"], list)
    assert "num_detections" in data
    assert isinstance(data["num_detections"], int)
    assert data["num_detections"] >= 0
    assert data["num_detections"] == len(data["detections"])

    # Verify detection structure if any exist
    for detection in data["detections"]:
        assert "label" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        assert isinstance(detection["confidence"], (int, float))
        assert 0 <= detection["confidence"] <= 1
        assert isinstance(detection["bbox"], list)
        assert len(detection["bbox"]) == 4


def test_moondream_point_response_structure(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test Moondream point response has correct structure.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/moondream/point",
        json={
            "file_id": uploaded_file_id,
            "object_description": "center",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert data["status"] == "success"
    assert data["object_description"] == "center"
    assert "points" in data
    assert isinstance(data["points"], list)
    assert "num_points" in data
    assert isinstance(data["num_points"], int)
    assert data["num_points"] >= 0
    assert data["num_points"] == len(data["points"])

    # Verify point structure if any exist
    for point in data["points"]:
        assert "x" in point
        assert "y" in point
        assert isinstance(point["x"], (int, float))
        assert isinstance(point["y"], (int, float))
