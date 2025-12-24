"""Tests for SAM (Segment Anything Model) endpoints."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient


def test_sam_generate_endpoint(client: TestClient, sample_image: BytesIO) -> None:
    """Test SAM automatic mask generation endpoint.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/sam/generate",
        files={"file": ("test.png", sample_image, "image/png")},
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
    sample_image: BytesIO,
    points_per_side: int,
    pred_iou_thresh: float,
) -> None:
    """Test SAM generation with different parameters.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
        points_per_side: Grid sampling density
        pred_iou_thresh: IoU threshold
    """
    # Reset BytesIO position
    sample_image.seek(0)

    response = client.post(
        "/api/v1/sam/generate",
        files={"file": ("test.png", sample_image, "image/png")},
        data={
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_sam_predict_with_points(client: TestClient, sample_image: BytesIO) -> None:
    """Test SAM prompt-based prediction with point prompts.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/sam/predict",
        files={"file": ("test.png", sample_image, "image/png")},
        json={
            "point_coords": [[50, 50], [75, 75]],
            "point_labels": [1, 1],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "masks" in data
    assert "metadata" in data


def test_sam_predict_with_boxes(client: TestClient, sample_image: BytesIO) -> None:
    """Test SAM prediction with bounding box prompts.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/sam/predict",
        files={"file": ("test.png", sample_image, "image/png")},
        json={
            "boxes": [[10, 10, 90, 90]],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_sam_predict_without_prompts(client: TestClient, sample_image: BytesIO) -> None:
    """Test SAM prediction fails without any prompts.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/sam/predict",
        files={"file": ("test.png", sample_image, "image/png")},
    )

    # Should return 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_sam_batch_processing(client: TestClient, sample_image: BytesIO) -> None:
    """Test SAM batch processing endpoint.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    # Create multiple file instances
    sample_image.seek(0)
    image1 = BytesIO(sample_image.read())
    sample_image.seek(0)
    image2 = BytesIO(sample_image.read())

    response = client.post(
        "/api/v1/sam/batch",
        files=[
            ("files", ("test1.png", image1, "image/png")),
            ("files", ("test2.png", image2, "image/png")),
        ],
        json={
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
