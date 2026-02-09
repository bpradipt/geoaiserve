"""FastAPI dependencies for inference concurrency control and API key validation."""

from __future__ import annotations

import threading
from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from .config import get_settings

settings = get_settings()
_inference_semaphore = threading.Semaphore(settings.max_concurrent_inference)


def require_inference_slot() -> Generator[None, None, None]:
    """FastAPI dependency that gates inference to a limited number of concurrent requests.

    Attempts to acquire the inference semaphore without blocking. If the semaphore
    cannot be acquired (another request is in progress), immediately raises HTTP 503.

    Yields:
        None when the slot is acquired

    Raises:
        HTTPException: 503 if no inference slot is available
    """
    acquired = _inference_semaphore.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is busy processing another request",
        )
    try:
        yield
    finally:
        _inference_semaphore.release()


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """Validate the API key if API key authentication is enabled.

    When ``settings.api_key_required`` is False (the default), this is a no-op.
    When True, the request must include a valid ``X-API-Key`` header whose value
    is present in ``settings.api_keys``, otherwise a 401 is raised.
    """
    settings = get_settings()
    if not settings.api_key_required:
        return
    if not api_key or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
