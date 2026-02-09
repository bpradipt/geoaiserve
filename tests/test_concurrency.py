"""Tests for inference concurrency control."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from geoaiserve.dependencies import _inference_semaphore

pytestmark = pytest.mark.mock


def test_single_request_succeeds(client: TestClient, uploaded_file_id: str) -> None:
    """A single inference request should succeed normally."""
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": uploaded_file_id},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_concurrent_request_rejected(client: TestClient, uploaded_file_id: str) -> None:
    """A second request while the semaphore is held should get 503."""
    # Simulate a request already in progress by acquiring the semaphore
    _inference_semaphore.acquire()
    try:
        response = client.post(
            "/api/v1/sam/generate",
            json={"file_id": uploaded_file_id},
        )
        assert response.status_code == 503
        assert "busy" in response.json()["detail"].lower()
    finally:
        _inference_semaphore.release()


def test_request_succeeds_after_previous_completes(
    client: TestClient, uploaded_file_id: str
) -> None:
    """After a held semaphore is released, the next request should succeed."""
    # Acquire and release to simulate a completed request
    _inference_semaphore.acquire()
    _inference_semaphore.release()

    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": uploaded_file_id},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_non_inference_endpoint_not_gated(client: TestClient) -> None:
    """Non-inference endpoints (health, files) should not be gated."""
    # Hold the semaphore to block inference endpoints
    _inference_semaphore.acquire()
    try:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    finally:
        _inference_semaphore.release()


def test_concurrent_request_from_thread(
    client: TestClient, uploaded_file_id: str
) -> None:
    """Test that a request from another thread is rejected while semaphore is held."""
    _inference_semaphore.acquire()
    result = {}

    def make_request():
        resp = client.post(
            "/api/v1/moondream/caption",
            json={"file_id": uploaded_file_id},
        )
        result["status_code"] = resp.status_code
        result["detail"] = resp.json().get("detail", "")

    try:
        t = threading.Thread(target=make_request)
        t.start()
        t.join(timeout=5)
        assert result["status_code"] == 503
        assert "busy" in result["detail"].lower()
    finally:
        _inference_semaphore.release()


def test_semaphore_released_on_error(client: TestClient) -> None:
    """The semaphore should be released even if the request errors out."""
    # Make a request that will fail (bad file_id) — semaphore should still be released
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": "nonexistent-file-id"},
    )
    assert response.status_code in (400, 404)

    # The semaphore should be available for the next request
    acquired = _inference_semaphore.acquire(blocking=False)
    assert acquired, "Semaphore was not released after failed request"
    _inference_semaphore.release()
