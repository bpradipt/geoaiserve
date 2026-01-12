"""Integration tests for cross-model file usage and lifecycle."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def test_upload_once_use_multiple_models(client: TestClient, sample_image: BytesIO) -> None:
    """Test that a single uploaded file can be used with multiple models.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    # Upload file once
    sample_image.seek(0)
    upload_response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    assert upload_response.status_code == 200
    file_id = upload_response.json()["file_id"]

    # Use with SAM
    sam_response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": file_id},
    )
    assert sam_response.status_code == 200
    assert sam_response.json()["status"] == "success"

    # Use with Moondream
    moondream_response = client.post(
        "/api/v1/moondream/caption",
        json={"file_id": file_id},
    )
    assert moondream_response.status_code == 200
    assert moondream_response.json()["status"] == "success"

    # Use with DINOv3
    dinov3_response = client.post(
        "/api/v1/dinov3/features",
        json={"file_id": file_id},
    )
    assert dinov3_response.status_code == 200
    assert dinov3_response.json()["status"] == "success"


def test_file_reuse_across_requests(client: TestClient, sample_image: BytesIO) -> None:
    """Test that a file can be used multiple times with the same model.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    # Upload file
    sample_image.seek(0)
    upload_response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    assert upload_response.status_code == 200
    file_id = upload_response.json()["file_id"]

    # Use file with SAM multiple times
    for i in range(3):
        response = client.post(
            "/api/v1/sam/generate",
            json={
                "file_id": file_id,
                "points_per_side": 16 + (i * 16),  # 16, 32, 48
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"


def test_model_request_after_file_deleted(client: TestClient, sample_image: BytesIO) -> None:
    """Test that model requests fail after file is deleted.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    # Upload file
    sample_image.seek(0)
    upload_response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    assert upload_response.status_code == 200
    file_id = upload_response.json()["file_id"]

    # Verify file works with model
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": file_id},
    )
    assert response.status_code == 200

    # Delete file
    delete_response = client.delete(f"/api/v1/files/{file_id}")
    assert delete_response.status_code == 200

    # Try to use deleted file - should return 404
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": file_id},
    )
    assert response.status_code == 404


def test_different_image_sizes_across_models(client: TestClient) -> None:
    """Test that different image sizes work across models."""
    sizes = [(50, 50), (100, 100), (200, 200)]

    for width, height in sizes:
        # Create image of specific size
        img = Image.new("RGB", (width, height), color="blue")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Upload
        upload_response = client.post(
            "/api/v1/files/upload",
            files={"file": (f"test_{width}x{height}.png", img_bytes, "image/png")},
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        # Test with each model
        sam_response = client.post(
            "/api/v1/sam/generate",
            json={"file_id": file_id, "points_per_side": 16},
        )
        assert sam_response.status_code == 200

        caption_response = client.post(
            "/api/v1/moondream/caption",
            json={"file_id": file_id},
        )
        assert caption_response.status_code == 200

        features_response = client.post(
            "/api/v1/dinov3/features",
            json={"file_id": file_id},
        )
        assert features_response.status_code == 200


def test_concurrent_model_usage(
    client: TestClient,
    uploaded_file_ids: list[str],
) -> None:
    """Test using different files with different models.

    Args:
        client: FastAPI test client
        uploaded_file_ids: List of uploaded file IDs
    """
    # Use first file with SAM
    sam_response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": uploaded_file_ids[0]},
    )
    assert sam_response.status_code == 200

    # Use second file with Moondream
    moondream_response = client.post(
        "/api/v1/moondream/caption",
        json={"file_id": uploaded_file_ids[1]},
    )
    assert moondream_response.status_code == 200

    # Use third file with DINOv3
    dinov3_response = client.post(
        "/api/v1/dinov3/features",
        json={"file_id": uploaded_file_ids[2]},
    )
    assert dinov3_response.status_code == 200

    # All files should still exist
    for file_id in uploaded_file_ids:
        response = client.get(f"/api/v1/files/{file_id}")
        assert response.status_code == 200


def test_batch_operations_with_uploaded_files(
    client: TestClient,
    uploaded_file_ids: list[str],
) -> None:
    """Test batch operations using multiple uploaded files.

    Args:
        client: FastAPI test client
        uploaded_file_ids: List of uploaded file IDs
    """
    # SAM batch processing
    sam_batch_response = client.post(
        "/api/v1/sam/batch",
        json={
            "file_ids": uploaded_file_ids[:2],
            "predict_params": {
                "point_coords": [[50, 50]],
                "point_labels": [1],
            },
        },
    )
    assert sam_batch_response.status_code == 200
    assert sam_batch_response.json()["total_images"] == 2

    # DINOv3 batch similarity
    dinov3_batch_response = client.post(
        "/api/v1/dinov3/batch-similarity",
        json={
            "query_file_id": uploaded_file_ids[0],
            "candidate_file_ids": uploaded_file_ids[1:],
        },
    )
    assert dinov3_batch_response.status_code == 200
    assert dinov3_batch_response.json()["num_candidates"] == 2
