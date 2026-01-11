"""Tests for SAM (Segment Anything Model) endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


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
