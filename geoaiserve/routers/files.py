"""File upload and management API endpoints."""

from __future__ import annotations

import io
import logging

import numpy as np
import rasterio
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from ..schemas.files import (
    FileDeleteResponse,
    FileInfo,
    FileListResponse,
    UploadResponse,
)
from ..services.file_handler import file_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["Files"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a file",
    description="Upload an image file for processing. Returns a file_id to use with model endpoints.",
)
async def upload_file(
    file: UploadFile = File(..., description="Image file to upload"),
) -> UploadResponse:
    """Upload a file and return file_id for later use with model endpoints."""
    file_id, file_path = await file_handler.save_upload_with_id(file)
    stat = file_path.stat()

    logger.info(f"File uploaded: {file_id} ({file.filename})")

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or file_path.name,
        size=stat.st_size,
        content_type=file.content_type or "application/octet-stream",
    )


@router.get(
    "/{file_id}/preview",
    summary="Get image preview",
    description="Get a PNG preview of the uploaded image. Useful for displaying GeoTIFF files in browsers.",
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "File not found"},
    },
)
async def get_file_preview(file_id: str, max_size: int = 800) -> Response:
    """Generate a PNG preview of an uploaded image.

    Args:
        file_id: The file ID from upload
        max_size: Maximum dimension (width or height) for the preview

    Returns:
        PNG image response
    """
    try:
        file_path = file_handler.get_file_by_id(file_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    try:
        # Try to open with rasterio (works for GeoTIFF and other raster formats)
        with rasterio.open(file_path) as src:
            # Read first 3 bands (or less if fewer available)
            num_bands = min(3, src.count)
            bands = [src.read(i + 1) for i in range(num_bands)]

            # Stack bands and normalize to 0-255
            if num_bands == 1:
                # Grayscale
                arr = bands[0]
            else:
                # RGB
                arr = np.stack(bands, axis=-1)

            # Handle different data types and normalize
            if arr.dtype != np.uint8:
                # Normalize to 0-255
                arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
                if arr_max > arr_min:
                    arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)

            # Handle NaN values
            arr = np.nan_to_num(arr, nan=0).astype(np.uint8)

            # Create PIL image
            if num_bands == 1:
                img = Image.fromarray(arr, mode='L')
            else:
                img = Image.fromarray(arr, mode='RGB')

            # Resize if needed
            if img.width > max_size or img.height > max_size:
                ratio = max_size / max(img.width, img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to PNG bytes
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)

            return Response(
                content=buffer.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"inline; filename={file_id}.png",
                    "Cache-Control": "max-age=3600",
                }
            )

    except Exception as e:
        logger.error(f"Failed to generate preview for {file_id}: {e}")
        # Try with PIL directly for standard image formats
        try:
            img = Image.open(file_path)
            if img.width > max_size or img.height > max_size:
                ratio = max_size / max(img.width, img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)

            return Response(
                content=buffer.getvalue(),
                media_type="image/png",
            )
        except Exception as e2:
            logger.error(f"PIL fallback also failed: {e2}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate preview: {str(e)}"
            )


@router.get(
    "/{file_id}",
    response_model=FileInfo,
    summary="Get file metadata",
    description="Get metadata for an uploaded file by file_id.",
)
async def get_file_info(file_id: str) -> FileInfo:
    """Get file metadata by file_id."""
    info = file_handler.get_file_info(file_id)
    return FileInfo(**info)


@router.delete(
    "/{file_id}",
    response_model=FileDeleteResponse,
    summary="Delete uploaded file",
    description="Delete an uploaded file by file_id.",
)
async def delete_file(file_id: str) -> FileDeleteResponse:
    """Delete an uploaded file."""
    deleted = file_handler.delete_file_by_id(file_id)
    logger.info(f"File deleted: {file_id}")
    return FileDeleteResponse(file_id=file_id, deleted=deleted)


@router.get(
    "/",
    response_model=FileListResponse,
    summary="List uploaded files",
    description="List all uploaded files.",
)
async def list_files() -> FileListResponse:
    """List all uploaded files."""
    files = file_handler.list_files()
    return FileListResponse(
        files=[FileInfo(**f) for f in files],
        total=len(files),
    )
