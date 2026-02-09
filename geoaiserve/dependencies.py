"""FastAPI dependencies for inference concurrency control."""

from __future__ import annotations

import threading
from collections.abc import Generator

from fastapi import HTTPException, status

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
