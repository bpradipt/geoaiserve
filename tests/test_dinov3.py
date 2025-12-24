"""Tests for DINOv3 feature extraction and similarity endpoints."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient


def test_dinov3_extract_features(client: TestClient, sample_image: BytesIO) -> None:
    """Test DINOv3 feature extraction endpoint.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/dinov3/features",
        files={"file": ("test.png", sample_image, "image/png")},
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
    sample_image: BytesIO,
    return_patch_features: bool,
) -> None:
    """Test DINOv3 feature extraction with/without patch features.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
        return_patch_features: Whether to return patch features
    """
    sample_image.seek(0)

    response = client.post(
        "/api/v1/dinov3/features",
        files={"file": ("test.png", sample_image, "image/png")},
        json={"return_patch_features": return_patch_features},
    )

    assert response.status_code == 200
    data = response.json()

    if return_patch_features:
        # Patch features may be None for mock model
        assert "patch_features" in data
    else:
        assert data.get("patch_features") is None


def test_dinov3_similarity(client: TestClient, sample_image: BytesIO) -> None:
    """Test DINOv3 patch similarity computation.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    response = client.post(
        "/api/v1/dinov3/similarity",
        files={"file": ("test.png", sample_image, "image/png")},
        json={
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
    sample_image: BytesIO,
    query_points: list[list[float]],
) -> None:
    """Test DINOv3 similarity with different numbers of query points.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
        query_points: List of query points
    """
    sample_image.seek(0)

    response = client.post(
        "/api/v1/dinov3/similarity",
        files={"file": ("test.png", sample_image, "image/png")},
        json={"query_points": query_points},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["similarity_maps"]) == len(query_points)


def test_dinov3_batch_similarity(client: TestClient, sample_image: BytesIO) -> None:
    """Test DINOv3 batch similarity ranking.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
    """
    # Create multiple file instances
    sample_image.seek(0)
    query = BytesIO(sample_image.read())

    sample_image.seek(0)
    candidate1 = BytesIO(sample_image.read())

    sample_image.seek(0)
    candidate2 = BytesIO(sample_image.read())

    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        files=[
            ("query_file", ("query.png", query, "image/png")),
            ("candidate_files", ("cand1.png", candidate1, "image/png")),
            ("candidate_files", ("cand2.png", candidate2, "image/png")),
        ],
        json={"top_k": 10},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["num_candidates"] == 2
    assert "similarities" in data
    assert isinstance(data["similarities"], list)
    assert len(data["similarities"]) <= 10  # Respects top_k


@pytest.mark.parametrize(
    "top_k",
    [1, 5, 10],
)
def test_dinov3_batch_similarity_top_k(
    client: TestClient,
    sample_image: BytesIO,
    top_k: int,
) -> None:
    """Test DINOv3 batch similarity with different top_k values.

    Args:
        client: FastAPI test client
        sample_image: Sample test image
        top_k: Number of top results to return
    """
    sample_image.seek(0)
    query = BytesIO(sample_image.read())

    candidates = []
    for _ in range(5):
        sample_image.seek(0)
        candidates.append(BytesIO(sample_image.read()))

    files = [("query_file", ("query.png", query, "image/png"))]
    for i, cand in enumerate(candidates):
        files.append(("candidate_files", (f"cand{i}.png", cand, "image/png")))

    response = client.post(
        "/api/v1/dinov3/batch-similarity",
        files=files,
        json={"top_k": top_k},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["similarities"]) <= top_k
