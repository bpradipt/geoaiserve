"""API routers for different model endpoints."""

from .common import router as common_router
from .dinov3 import router as dinov3_router
from .moondream import router as moondream_router
from .sam import router as sam_router

__all__ = [
    "common_router",
    "dinov3_router",
    "moondream_router",
    "sam_router",
]
