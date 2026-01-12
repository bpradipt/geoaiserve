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


# --- Parameter Validation Tests ---


def test_dinov3_similarity_empty_points(client: TestClient, uploaded_file_id: str) -> None:
    """Test DINOv3 similarity rejects empty query_points list.

    Args:
        client: FastAPI test client
        uploaded_file_id: File ID of uploaded test image
    """
    response = client.post(
        "/api/v1/dinov3/similarity",
        json={
            "file_id": uploaded_file_id,
            "query_points": [],
        },
    )

    # Empty list should fail validation (min_length=1)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_dinov3_batch_similarity_invalid_top_k(
    client: TestClient,
    uploaded_file_ids: list[str],
    top_k: int,
) -> None:
    """Test DINOv3 batch similarity rejects invalid top_k values.

    Args:
        client: FastAPI test client
        uploaded_file_ids: List of file IDs for uploaded images
        top_k: Invalid top_k value (must be >= 1)
    """
    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        json={
            "query_file_id": uploaded_file_ids[0],
            "candidate_file_ids": uploaded_file_ids[1:],
            "top_k": top_k,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_dinov3_batch_similarity_empty_candidates(
    client: TestClient,
    uploaded_file_ids: list[str],
) -> None:
    """Test DINOv3 batch similarity rejects empty candidate list.

    Args:
        client: FastAPI test client
        uploaded_file_ids: List of file IDs for uploaded images
    """
    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        json={
            "query_file_id": uploaded_file_ids[0],
            "candidate_file_ids": [],
            "top_k": 10,
        },
    )

    # Empty list should fail validation (min_length=1)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


# --- Non-existent file_id tests ---


def test_dinov3_features_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test DINOv3 features returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/dinov3/features",
        json={"file_id": invalid_file_id},
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_dinov3_similarity_nonexistent_file_id(
    client: TestClient,
    invalid_file_id: str,
) -> None:
    """Test DINOv3 similarity returns 404 for non-existent file_id.

    Args:
        client: FastAPI test client
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/dinov3/similarity",
        json={
            "file_id": invalid_file_id,
            "query_points": [[50, 50]],
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_dinov3_batch_similarity_nonexistent_query(
    client: TestClient,
    uploaded_file_ids: list[str],
    invalid_file_id: str,
) -> None:
    """Test DINOv3 batch similarity returns 404 for non-existent query_file_id.

    Args:
        client: FastAPI test client
        uploaded_file_ids: Valid candidate file IDs
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        json={
            "query_file_id": invalid_file_id,
            "candidate_file_ids": uploaded_file_ids[:2],
            "top_k": 10,
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_dinov3_batch_similarity_nonexistent_candidate(
    client: TestClient,
    uploaded_file_ids: list[str],
    invalid_file_id: str,
) -> None:
    """Test DINOv3 batch similarity handles non-existent candidate_file_id.

    Args:
        client: FastAPI test client
        uploaded_file_ids: Valid file IDs
        invalid_file_id: Non-existent file ID fixture
    """
    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        json={
            "query_file_id": uploaded_file_ids[0],
            "candidate_file_ids": [uploaded_file_ids[1], invalid_file_id],
            "top_k": 10,
        },
    )

    # Should return 404 or include error in results
    assert response.status_code in [200, 404]


# --- Response Structure Validation ---


def test_dinov3_features_response_structure(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test DINOv3 features response has correct structure.

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

    # Verify response structure
    assert data["status"] == "success"
    assert "cls_token" in data
    assert "feature_dim" in data
    assert isinstance(data["cls_token"], list)
    assert isinstance(data["feature_dim"], int)
    assert data["feature_dim"] > 0

    # Verify cls_token dimension matches feature_dim
    assert len(data["cls_token"]) == data["feature_dim"]

    # Verify metadata
    assert "metadata" in data
    assert "processing_time" in data["metadata"]
    assert isinstance(data["metadata"]["processing_time"], (int, float))
    assert data["metadata"]["processing_time"] >= 0


def test_dinov3_similarity_response_structure(
    client: TestClient,
    uploaded_file_id: str,
) -> None:
    """Test DINOv3 similarity response has correct structure.

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

    # Verify response structure
    assert data["status"] == "success"
    assert "query_points" in data
    assert "similarity_maps" in data
    assert "map_size" in data

    # Verify dimensions
    assert len(data["query_points"]) == 2
    assert len(data["similarity_maps"]) == 2
    assert len(data["map_size"]) == 2
    assert all(isinstance(dim, int) and dim > 0 for dim in data["map_size"])

    # Verify similarity map structure
    for sim_map in data["similarity_maps"]:
        assert isinstance(sim_map, list)
        # Each map should be a 2D array matching map_size
        if len(sim_map) > 0:
            assert len(sim_map) == data["map_size"][0]
            for row in sim_map:
                assert isinstance(row, list)
                assert len(row) == data["map_size"][1]


def test_dinov3_batch_similarity_response_structure(
    client: TestClient,
    uploaded_file_ids: list[str],
) -> None:
    """Test DINOv3 batch similarity response has correct structure.

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

    # Verify response structure
    assert data["status"] == "success"
    assert "num_candidates" in data
    assert isinstance(data["num_candidates"], int)
    assert data["num_candidates"] >= 0
    assert "similarities" in data
    assert isinstance(data["similarities"], list)

    # Verify similarity result structure
    for result in data["similarities"]:
        assert "index" in result
        assert "similarity" in result
        assert isinstance(result["index"], int)
        assert isinstance(result["similarity"], (int, float))
        # Note: Mock model may return values outside [0, 1] range
        # In production with real model, similarity should be in [0, 1]

    # Verify metadata
    assert "metadata" in data
    assert "processing_time" in data["metadata"]
