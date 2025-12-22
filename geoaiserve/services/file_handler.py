"""File upload and download handling service."""

from __future__ import annotations

import base64
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import httpx
import rasterio
from fastapi import HTTPException, UploadFile, status
from rasterio.io import MemoryFile
from starlette.responses import FileResponse, StreamingResponse

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file uploads, downloads, and storage operations."""

    def __init__(self, settings: Settings | None = None):
        """Initialize file handler.

        Args:
            settings: Application settings (uses default if not provided)
        """
        self.settings = settings or get_settings()
        self.storage_path = self.settings.storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def save_upload(
        self,
        file: UploadFile,
        validate_image: bool = True
    ) -> Path:
        """Save uploaded file to storage.

        Args:
            file: Uploaded file
            validate_image: Whether to validate as image file

        Returns:
            Path to saved file

        Raises:
            HTTPException: If file validation fails
        """
        # Validate content type
        if validate_image and file.content_type not in self.settings.allowed_image_formats:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {file.content_type}. "
                       f"Allowed types: {self.settings.allowed_image_formats}"
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename or "").suffix or ".tif"
        file_path = self.storage_path / f"{file_id}{file_ext}"

        # Read and validate file size
        content = await file.read()
        if len(content) > self.settings.max_upload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {self.settings.max_upload_size} bytes"
            )

        # Save file
        try:
            file_path.write_bytes(content)
            logger.info(f"Saved uploaded file to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}"
            )

    async def fetch_from_url(self, url: str) -> Path:
        """Fetch file from URL and save to storage.

        Args:
            url: URL to fetch from

        Returns:
            Path to saved file

        Raises:
            HTTPException: If fetch fails
        """
        file_id = str(uuid.uuid4())
        file_path = self.storage_path / f"{file_id}.tif"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Check file size
                if len(response.content) > self.settings.max_upload_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Max size: {self.settings.max_upload_size} bytes"
                    )

                file_path.write_bytes(response.content)
                logger.info(f"Fetched file from {url} to {file_path}")
                return file_path

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch file from {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch file from URL: {str(e)}"
            )

    def decode_base64(self, base64_data: str) -> Path:
        """Decode base64 data and save to storage.

        Args:
            base64_data: Base64 encoded file data

        Returns:
            Path to saved file

        Raises:
            HTTPException: If decode fails
        """
        file_id = str(uuid.uuid4())
        file_path = self.storage_path / f"{file_id}.tif"

        try:
            # Remove data URL prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]

            content = base64.b64decode(base64_data)

            # Check file size
            if len(content) > self.settings.max_upload_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Max size: {self.settings.max_upload_size} bytes"
                )

            file_path.write_bytes(content)
            logger.info(f"Decoded base64 data to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to decode base64 data: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode base64 data: {str(e)}"
            )

    def validate_geotiff(self, file_path: Path) -> dict[str, any]:
        """Validate and extract metadata from GeoTIFF file.

        Args:
            file_path: Path to GeoTIFF file

        Returns:
            Dictionary containing metadata

        Raises:
            HTTPException: If validation fails
        """
        try:
            with rasterio.open(file_path) as src:
                metadata = {
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": str(src.dtypes[0]),
                    "crs": str(src.crs) if src.crs else None,
                    "bounds": list(src.bounds),
                    "transform": list(src.transform)[:6] if src.transform else None,
                }
                logger.info(f"Validated GeoTIFF: {file_path}")
                return metadata

        except Exception as e:
            logger.error(f"Failed to validate GeoTIFF {file_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GeoTIFF file: {str(e)}"
            )

    def create_download_response(
        self,
        file_path: Path,
        filename: str | None = None,
        as_attachment: bool = True
    ) -> FileResponse:
        """Create a file download response.

        Args:
            file_path: Path to file to download
            filename: Optional custom filename
            as_attachment: Whether to force download vs inline display

        Returns:
            FastAPI FileResponse
        """
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )

        filename = filename or file_path.name
        media_type = self._get_media_type(file_path.suffix)

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type,
            headers={
                "Content-Disposition": f"{'attachment' if as_attachment else 'inline'}; filename={filename}"
            }
        )

    def create_streaming_response(
        self,
        file_path: Path,
        chunk_size: int = 8192
    ) -> StreamingResponse:
        """Create a streaming response for large files.

        Args:
            file_path: Path to file to stream
            chunk_size: Size of chunks to stream

        Returns:
            FastAPI StreamingResponse
        """
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )

        def iterfile():
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk

        media_type = self._get_media_type(file_path.suffix)

        return StreamingResponse(
            iterfile(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={file_path.name}"
            }
        )

    def cleanup_file(self, file_path: Path) -> bool:
        """Delete a file from storage.

        Args:
            file_path: Path to file to delete

        Returns:
            True if deleted, False if file didn't exist
        """
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
            return True
        return False

    def _get_media_type(self, extension: str) -> str:
        """Get media type for file extension.

        Args:
            extension: File extension (including dot)

        Returns:
            MIME type string
        """
        media_types = {
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".geotiff": "image/tiff",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".json": "application/json",
            ".geojson": "application/geo+json",
            ".shp": "application/x-shapefile",
            ".zip": "application/zip",
            ".gpkg": "application/geopackage+sqlite3",
            ".kml": "application/vnd.google-earth.kml+xml",
        }
        return media_types.get(extension.lower(), "application/octet-stream")


# Global file handler instance
file_handler = FileHandler()
