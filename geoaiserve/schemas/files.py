"""File upload and management schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response after successful file upload."""

    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FileInfo(BaseModel):
    """File metadata information."""

    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    created_at: datetime = Field(..., description="Upload timestamp")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")
    path: str = Field(..., description="Storage path")


class FileListResponse(BaseModel):
    """Response for listing uploaded files."""

    files: list[FileInfo] = Field(default_factory=list, description="List of uploaded files")
    total: int = Field(..., description="Total number of files")


class FileDeleteResponse(BaseModel):
    """Response after file deletion."""

    file_id: str = Field(..., description="Deleted file identifier")
    deleted: bool = Field(..., description="Whether deletion was successful")
