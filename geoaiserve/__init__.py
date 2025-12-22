"""GeoAI REST API - FastAPI service for geospatial AI models."""

__version__ = "0.1.0"

from .main import app, create_app

__all__ = [
    "app",
    "create_app",
]
