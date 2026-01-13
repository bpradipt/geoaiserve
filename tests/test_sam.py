"""Tests for SAM (Segment Anything Model) endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.mock


def test_sam_generate_endpoint(client: TestClient, uploaded_file_id: str) -> None:
    """Test SAM automatic mask generation endpoint.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": uploaded_file_id},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "num_masks" in data
    assert "masks" in data
    assert "metadata" in data
    assert "processing_time" in data["metadata"]
    assert "model_name" in data["metadata"]


@pytest.mark.parametrize(
    ("points_per_side", "pred_iou_thresh"),
    [
        (16, 0.88),
        (32, 0.90),
        (64, 0.85),
    ],
)
def test_sam_generate_with_params(
    client: TestClient,
    uploaded_file_id: str,
    points_per_side: int,
    pred_iou_thresh: float,
) -> None:
    """Test SAM generation with different parameters.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        points_per_side: Grid sampling density
        pred_iou_thresh: IoU threshold
    """
    response = client.post(
        "/api/v1/sam/generate",
        json={
            "file_id": uploaded_file_id,
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_sam_predict_with_points(client: TestClient, uploaded_file_id: str) -> None:
    """Test SAM prompt-based prediction with point prompts.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/sam/predict",
        json={
            "file_id": uploaded_file_id,
            "point_coords": [[50, 50], [75, 75]],
            "point_labels": [1, 1],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "masks" in data
    assert "metadata" in data


def test_sam_predict_with_boxes(client: TestClient, uploaded_file_id: str) -> None:
    """Test SAM prediction with bounding box prompts.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/sam/predict",
        json={
            "file_id": uploaded_file_id,
            "boxes": [[10, 10, 90, 90]],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_sam_predict_without_prompts(client: TestClient, uploaded_file_id: str) -> None:
    """Test SAM prediction fails without any prompts.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/sam/predict",
        json={"file_id": uploaded_file_id},
    )

    # Should return 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_sam_predict_without_file_id(client: TestClient) -> None:
    """Test SAM prediction fails without file_id.

    Args:
        client: FastAPI test client
    """
    response = client.post(
        "/api/v1/sam/predict",
        json={
            "point_coords": [[50, 50]],
            "point_labels": [1],
        },
    )

    # Should return 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_sam_batch_processing(client: TestClient, uploaded_file_ids: list[str]) -> None:
    """Test SAM batch processing endpoint.

    Args:
        client: FastAPI test client
        uploaded_file_ids: List of file IDs for uploaded images
    """
    response = client.post(
        "/api/v1/sam/batch",
        json={
            "file_ids": uploaded_file_ids[:2],
            "batch_size": 4,
            "predict_params": {
                "point_coords": [[50, 50]],
                "point_labels": [1],
            },
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["total_images"] == 2
    assert "results" in data
    assert len(data["results"]) <= 2


# --- Parameter Validation Tests ---


@pytest.mark.parametrize("points_per_side", [0, -1, 65, 100])
def test_sam_generate_invalid_points_per_side(
    client: TestClient,
    uploaded_file_id: str,
    points_per_side: int,
) -> None:
    """Test SAM generate rejects invalid points_per_side values.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        points_per_side: Invalid points_per_side value (must be 1-64)
    """
    response = client.post(
        "/api/v1/sam/generate",
        json={
            "file_id": uploaded_file_id,
            "points_per_side": points_per_side,
        },
    )

    # Should return 422 Unprocessable Entity for validation error
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.parametrize("pred_iou_thresh", [-0.1, 1.5, -1.0, 2.0])
def test_sam_generate_invalid_iou_thresh(
    client: TestClient,
    uploaded_file_id: str,
    pred_iou_thresh: float,
) -> None:
    """Test SAM generate rejects invalid pred_iou_thresh values.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        pred_iou_thresh: Invalid threshold (must be 0.0-1.0)
    """
    response = client.post(
        "/api/v1/sam/generate",
        json={
            "file_id": uploaded_file_id,
            "pred_iou_thresh": pred_iou_thresh,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.parametrize("stability_score_thresh", [-0.5, 1.1])
def test_sam_generate_invalid_stability_thresh(
    client: TestClient,
    uploaded_file_id: str,
    stability_score_thresh: float,
) -> None:
    """Test SAM generate rejects invalid stability_score_thresh values.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        stability_score_thresh: Invalid threshold (must be 0.0-1.0)
    """
    response = client.post(
        "/api/v1/sam/generate",
        json={
            "file_id": uploaded_file_id,
            "stability_score_thresh": stability_score_thresh,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_sam_generate_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test SAM generate returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": invalid_file_id},
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_sam_predict_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test SAM predict returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/sam/predict",
        json={
            "file_id": invalid_file_id,
            "point_coords": [[50, 50]],
            "point_labels": [1],
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_sam_predict_mismatched_coords_labels(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test SAM predict with mismatched point_coords and point_labels lengths.

    Note: This tests the API's handling of inconsistent input.
    The server may accept this (and the model handles it) or reject it.
    """
    response = client.post(
        "/api/v1/sam/predict",
        json={
            "file_id": uploaded_file_id,
            "point_coords": [[50, 50], [75, 75], [25, 25]],
            "point_labels": [1],  # Only 1 label for 3 coords
        },
    )

    # May return 400/422 for validation error, or 200 if model handles it
    assert response.status_code in [200, 400, 422, 500]


def test_sam_batch_nonexistent_file_id(
    client: TestClient,
    uploaded_file_ids: list[str],
    invalid_file_id: str,
) -> None:
    """Test SAM batch with one non-existent file_id.

    Args:
        client: FastAPI test client
        uploaded_file_ids: Valid file IDs
        invalid_file_id: Non-existent file ID
    """
    response = client.post(
        "/api/v1/sam/batch",
        json={
            "file_ids": [uploaded_file_ids[0], invalid_file_id],
            "predict_params": {
                "point_coords": [[50, 50]],
                "point_labels": [1],
            },
        },
    )

    # Should return 404 or include error in results
    assert response.status_code in [200, 404]


def test_sam_batch_empty_file_ids(client: TestClient) -> None:
    """Test SAM batch with empty file_ids list."""
    response = client.post(
        "/api/v1/sam/batch",
        json={
            "file_ids": [],
            "predict_params": {
                "point_coords": [[50, 50]],
                "point_labels": [1],
            },
        },
    )

    # Empty list should fail validation (min_length=1)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


# --- Response Structure Validation ---


def test_sam_generate_response_structure(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test SAM generate response has correct structure.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": uploaded_file_id},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert data["status"] == "success"
    assert isinstance(data["num_masks"], int)
    assert data["num_masks"] >= 0
    assert isinstance(data["masks"], list)
    assert data["num_masks"] == len(data["masks"])

    # Verify metadata structure
    assert "metadata" in data
    assert "processing_time" in data["metadata"]
    assert "model_name" in data["metadata"]
    assert isinstance(data["metadata"]["processing_time"], (int, float))
    assert data["metadata"]["processing_time"] >= 0

    # Verify mask structure if any masks exist
    for mask in data["masks"]:
        assert "mask_id" in mask
        assert "score" in mask
        assert "area" in mask
        assert "bbox" in mask
        assert isinstance(mask["mask_id"], int)
        assert isinstance(mask["score"], (int, float))
        assert 0 <= mask["score"] <= 1
        assert isinstance(mask["area"], int)
        assert mask["area"] >= 0
        assert isinstance(mask["bbox"], list)
        assert len(mask["bbox"]) == 4
