"""File upload and management API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, UploadFile

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
