"""Tests for common API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint returns correct status."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], list)


def test_list_models(client: TestClient) -> None:
    """Test models listing endpoint returns all models."""
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert "total" in data
    assert data["total"] == 3  # SAM, Moondream, DINOv3

    # Check all expected models are present
    model_ids = {model["model_id"] for model in data["models"]}
    assert model_ids == {"sam", "moondream", "dinov3"}


@pytest.mark.parametrize(
    ("model_id", "expected_tasks"),
    [
        ("sam", ["automatic_mask_generation", "prompt_based_segmentation"]),
        ("moondream", ["image_captioning", "visual_qa", "object_detection", "point_detection"]),
        ("dinov3", ["feature_extraction", "patch_similarity", "batch_similarity"]),
    ],
)
def test_get_model_info(client: TestClient, model_id: str, expected_tasks: list[str]) -> None:
    """Test getting information for specific models.

    Args:
        client: FastAPI test client
        model_id: Model identifier to query
        expected_tasks: Expected supported tasks for the model
    """
    response = client.get(f"/api/v1/models/{model_id}/info")

    assert response.status_code == 200
    data = response.json()

    assert data["model_id"] == model_id
    assert data["model_type"] == model_id
    assert "model_name" in data
    assert "description" in data
    assert data["supported_tasks"] == expected_tasks
    assert data["device"] in ["cpu", "cuda", "mps"]
    assert isinstance(data["loaded"], bool)


def test_get_invalid_model_info(client: TestClient) -> None:
    """Test getting info for non-existent model returns 404."""
    response = client.get("/api/v1/models/invalid-model/info")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_openapi_schema(client: TestClient) -> None:
    """Test OpenAPI schema is available."""
    response = client.get("/api/v1/openapi.json")

    assert response.status_code == 200
    data = response.json()

    assert "openapi" in data
    assert "info" in data
    assert "paths" in data

    # Check that all our endpoints are in the schema
    assert "/api/v1/health" in data["paths"]
    assert "/api/v1/models" in data["paths"]
    assert "/api/v1/sam/generate" in data["paths"]
    assert "/api/v1/moondream/caption" in data["paths"]
    assert "/api/v1/dinov3/features" in data["paths"]
