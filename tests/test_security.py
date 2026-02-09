"""Tests for API key validation and CORS configuration."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from geoaiserve.config import Settings

pytestmark = pytest.mark.mock

VALID_KEY = "test-key-abc"
VALID_KEY_2 = "test-key-xyz"


@contextmanager
def _override_settings(**overrides):
    """Patch get_settings everywhere it's imported so overrides apply at request time."""
    settings = Settings(**overrides)
    with (
        patch("geoaiserve.config.get_settings", return_value=settings),
        patch("geoaiserve.dependencies.get_settings", return_value=settings),
        patch("geoaiserve.main.get_settings", return_value=settings),
    ):
        yield settings


def _make_app():
    """Import and call create_app (must be called inside an _override_settings context)."""
    from geoaiserve.main import create_app

    return create_app()


# ---- API key required ----


class TestApiKeyRequired:
    """Tests with api_key_required=True."""

    def test_missing_key_returns_401(self) -> None:
        with _override_settings(api_key_required=True, api_keys=[VALID_KEY, VALID_KEY_2]):
            client = TestClient(_make_app())
            response = client.get("/api/v1/health")
        assert response.status_code == 401
        assert "API key" in response.json()["detail"]

    def test_invalid_key_returns_401(self) -> None:
        with _override_settings(api_key_required=True, api_keys=[VALID_KEY, VALID_KEY_2]):
            client = TestClient(_make_app())
            response = client.get(
                "/api/v1/health",
                headers={"X-API-Key": "wrong-key"},
            )
        assert response.status_code == 401
        assert "API key" in response.json()["detail"]

    def test_valid_key_returns_200(self) -> None:
        with _override_settings(api_key_required=True, api_keys=[VALID_KEY, VALID_KEY_2]):
            client = TestClient(_make_app())
            response = client.get(
                "/api/v1/health",
                headers={"X-API-Key": VALID_KEY},
            )
        assert response.status_code == 200

    def test_second_valid_key_returns_200(self) -> None:
        with _override_settings(api_key_required=True, api_keys=[VALID_KEY, VALID_KEY_2]):
            client = TestClient(_make_app())
            response = client.get(
                "/api/v1/health",
                headers={"X-API-Key": VALID_KEY_2},
            )
        assert response.status_code == 200


# ---- API key not required (default) ----


class TestApiKeyNotRequired:
    """Tests with api_key_required=False (default)."""

    def test_no_key_passes_through(self) -> None:
        with _override_settings(api_key_required=False):
            client = TestClient(_make_app())
            response = client.get("/api/v1/health")
        assert response.status_code == 200


# ---- CORS validation ----


class TestCorsValidation:
    """Verify credentials=True + origins=['*'] gets corrected."""

    def test_credentials_with_wildcard_origins_corrected(self, caplog) -> None:
        """credentials=True + origins=['*'] should log a warning and set credentials=False."""
        with caplog.at_level(logging.WARNING):
            with _override_settings():  # defaults: cors_credentials=True, cors_origins=["*"]
                app = _make_app()
                client = TestClient(app)

        assert any("CORS misconfiguration" in msg for msg in caplog.messages)

        # Verify the middleware was configured with allow_credentials=False
        # by checking a preflight response doesn't include the credentials header
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-credentials") != "true"
