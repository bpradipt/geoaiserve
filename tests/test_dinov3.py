"""Tests for DINOv3 feature extraction and similarity endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_dinov3_extract_features(client: TestClient, uploaded_file_id: str) -> None:
    """Test DINOv3 feature extraction endpoint.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/dinov3/features",
        json={"file_id": uploaded_file_id},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "cls_token" in data
    assert "feature_dim" in data
    assert isinstance(data["cls_token"], list)
    assert data["feature_dim"] > 0
    assert "metadata" in data


@pytest.mark.parametrize(
    "return_patch_features",
    [True, False],
)
def test_dinov3_features_with_patches(
    client: TestClient,
    uploaded_file_id: str,
    return_patch_features: bool,
) -> None:
    """Test DINOv3 feature extraction with/without patch features.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        return_patch_features: Whether to return patch features
    """
    response = client.post(
        "/api/v1/dinov3/features",
        json={
            "file_id": uploaded_file_id,
            "return_patch_features": return_patch_features,
        },
    )

    assert response.status_code == 200
    data = response.json()

    if return_patch_features:
        # Patch features may be None for mock model
        assert "patch_features" in data
    else:
        assert data.get("patch_features") is None


def test_dinov3_similarity(client: TestClient, uploaded_file_id: str) -> None:
    """Test DINOv3 patch similarity computation.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/dinov3/similarity",
        json={
            "file_id": uploaded_file_id,
            "query_points": [[50, 50], [75, 75]],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "query_points" in data
    assert "similarity_maps" in data
    assert "map_size" in data
    assert len(data["query_points"]) == 2
    assert len(data["similarity_maps"]) == 2
    assert isinstance(data["map_size"], list)
    assert len(data["map_size"]) == 2


@pytest.mark.parametrize(
    "query_points",
    [
        [[10, 10]],
        [[25, 25], [50, 50]],
        [[10, 10], [30, 30], [50, 50], [70, 70]],
    ],
)
def test_dinov3_similarity_multiple_points(
    client: TestClient,
    uploaded_file_id: str,
    query_points: list[list[float]],
) -> None:
    """Test DINOv3 similarity with different numbers of query points.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
        query_points: List of query points
    """
    response = client.post(
        "/api/v1/dinov3/similarity",
        json={
            "file_id": uploaded_file_id,
            "query_points": query_points,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["similarity_maps"]) == len(query_points)


def test_dinov3_batch_similarity(client: TestClient, uploaded_file_ids: list[str]) -> None:
    """Test DINOv3 batch similarity ranking.

    Args:
        client: FastAPI test client
        uploaded_file_ids: List of file IDs for uploaded images
    """
    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        json={
            "query_file_id": uploaded_file_ids[0],
            "candidate_file_ids": uploaded_file_ids[1:],
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["num_candidates"] == 2
    assert "similarities" in data
    assert isinstance(data["similarities"], list)
    assert len(data["similarities"]) <= 10


@pytest.mark.parametrize(
    "top_k",
    [1, 5, 10],
)
def test_dinov3_batch_similarity_top_k(
    client: TestClient,
    uploaded_file_ids: list[str],
    top_k: int,
) -> None:
    """Test DINOv3 batch similarity with different top_k values.

    Args:
        client: FastAPI test client
        uploaded_file_ids: List of file IDs for uploaded images
        top_k: Number of top results to return
    """
    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        json={
            "query_file_id": uploaded_file_ids[0],
            "candidate_file_ids": uploaded_file_ids[1:],
            "top_k": top_k,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["similarities"]) <= top_k


def test_dinov3_features_without_file_id(client: TestClient) -> None:
    """Test DINOv3 features fails without file_id.

    Args:
        client: FastAPI test client
    """
    response = client.post(
        "/api/v1/dinov3/features",
        json={"return_patch_features": True},
    )

    # Should return 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
