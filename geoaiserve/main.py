"""Main FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .models import registry
from .routers import common_router
from .schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Configured models: {settings.geoai_models}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Storage path: {settings.storage_path}")

    # Startup: Models are loaded lazily on first request
    logger.info("Application started successfully")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down application")
    registry.unload_all()
    logger.info("All models unloaded")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        GeoAI REST API provides endpoints for geospatial AI model inference including:
        - SAM (Segment Anything Model) for segmentation
        - Moondream for vision-language tasks
        - DINOv3 for feature extraction and similarity
        - GroundedSAM for text-prompted segmentation
        - Detectron2 for instance segmentation
        - TIMM for classification and segmentation
        """,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    if settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=settings.cors_credentials,
            allow_methods=settings.cors_methods,
            allow_headers=settings.cors_headers,
        )
        logger.info(f"CORS enabled with origins: {settings.cors_origins}")

    # Include routers
    app.include_router(
        common_router,
        prefix=settings.api_prefix,
    )

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors.

        Args:
            request: FastAPI request
            exc: Validation error exception

        Returns:
            JSON error response
        """
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                detail=exc.errors()
            ).model_dump()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions.

        Args:
            request: FastAPI request
            exc: Exception

        Returns:
            JSON error response
        """
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                message="An internal error occurred",
                detail=str(exc) if settings.debug else None
            ).model_dump()
        )

    logger.info(f"FastAPI app created with prefix: {settings.api_prefix}")
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "geoaiserve.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level,
    )
