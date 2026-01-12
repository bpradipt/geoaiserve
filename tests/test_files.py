"""Tests for file upload and management endpoints."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient


def test_upload_file(client: TestClient, sample_image: BytesIO) -> None:
    """Test file upload endpoint."""
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "file_id" in data
    assert data["filename"] == "test.png"
    assert data["size"] > 0
    assert data["content_type"] == "image/png"
    assert "created_at" in data


def test_upload_file_jpeg(client: TestClient) -> None:
    """Test file upload with JPEG format."""
    from PIL import Image

    # Create JPEG image
    img = Image.new("RGB", (50, 50), color="green")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["content_type"] == "image/jpeg"


def test_get_file_info(client: TestClient, sample_image: BytesIO) -> None:
    """Test getting file metadata."""
    # First upload
    upload_response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    file_id = upload_response.json()["file_id"]

    # Then get info
    response = client.get(f"/api/v1/files/{file_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["file_id"] == file_id
    assert data["filename"].endswith(".png")
    assert data["size"] > 0
    assert "created_at" in data
    assert "expires_at" in data
    assert "path" in data


def test_get_nonexistent_file(client: TestClient) -> None:
    """Test getting info for non-existent file."""
    response = client.get("/api/v1/files/nonexistent-uuid-12345")
    assert response.status_code == 404


def test_delete_file(client: TestClient, sample_image: BytesIO) -> None:
    """Test file deletion."""
    # Upload
    sample_image.seek(0)
    upload_response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    file_id = upload_response.json()["file_id"]

    # Verify file exists
    response = client.get(f"/api/v1/files/{file_id}")
    assert response.status_code == 200

    # Delete
    response = client.delete(f"/api/v1/files/{file_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert response.json()["file_id"] == file_id

    # Verify deleted
    response = client.get(f"/api/v1/files/{file_id}")
    assert response.status_code == 404


def test_list_files(client: TestClient, sample_image: BytesIO) -> None:
    """Test listing uploaded files."""
    # Upload a file
    sample_image.seek(0)
    upload_response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test_list.png", sample_image, "image/png")},
    )
    uploaded_file_id = upload_response.json()["file_id"]

    # List files
    response = client.get("/api/v1/files/")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert "total" in data
    assert data["total"] >= 1

    # Check uploaded file is in list
    file_ids = [f["file_id"] for f in data["files"]]
    assert uploaded_file_id in file_ids


def test_upload_unsupported_type(client: TestClient) -> None:
    """Test uploading unsupported file type."""
    # Create a text file (not an image)
    text_content = BytesIO(b"This is not an image")

    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.txt", text_content, "text/plain")},
    )

    assert response.status_code == 415  # Unsupported Media Type


def test_upload_and_use_file_id(client: TestClient, sample_image: BytesIO) -> None:
    """Test uploading file and using file_id with model endpoint."""
    # Upload file
    sample_image.seek(0)
    upload_response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    assert upload_response.status_code == 200
    file_id = upload_response.json()["file_id"]

    # Use file_id with SAM generate endpoint
    response = client.post(
        "/api/v1/sam/generate",
        json={"file_id": file_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_multiple_uploads(client: TestClient, sample_image: BytesIO) -> None:
    """Test uploading multiple files."""
    file_ids = []

    for i in range(3):
        sample_image.seek(0)
        response = client.post(
            "/api/v1/files/upload",
            files={"file": (f"test_{i}.png", sample_image, "image/png")},
        )
        assert response.status_code == 200
        file_ids.append(response.json()["file_id"])

    # Verify all files exist
    for file_id in file_ids:
        response = client.get(f"/api/v1/files/{file_id}")
        assert response.status_code == 200

    # Verify in list
    response = client.get("/api/v1/files/")
    listed_ids = [f["file_id"] for f in response.json()["files"]]
    for file_id in file_ids:
        assert file_id in listed_ids


def test_delete_nonexistent_file(client: TestClient) -> None:
    """Test deleting a non-existent file returns 404."""
    response = client.delete("/api/v1/files/nonexistent-uuid-12345")
    assert response.status_code == 404


def test_upload_empty_file(client: TestClient) -> None:
    """Test uploading an empty file.

    Note: Currently the server accepts empty files. This test documents
    that behavior. If validation is added later, update this test.
    """
    empty_content = BytesIO(b"")

    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("empty.png", empty_content, "image/png")},
    )

    # Currently accepts empty files - update if validation is added
    assert response.status_code == 200
    data = response.json()
    assert data["size"] == 0


def test_upload_corrupted_image(client: TestClient, corrupted_image_bytes: BytesIO) -> None:
    """Test uploading corrupted image data is handled.

    Args:
        client: FastAPI test client
        corrupted_image_bytes: Invalid image bytes fixture
    """
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("bad.png", corrupted_image_bytes, "image/png")},
    )

    # Corrupted images should be rejected or handled gracefully
    # Accept either 400/415/422 for validation error, or 200 if server accepts raw bytes
    assert response.status_code in [200, 400, 415, 422]


def test_upload_special_characters_filename(client: TestClient, sample_image: BytesIO) -> None:
    """Test uploading file with special characters in filename."""
    sample_image.seek(0)
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("tëst_ímàgé_日本語.png", sample_image, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data


def test_upload_small_image(client: TestClient, sample_image_small: BytesIO) -> None:
    """Test uploading a 1x1 pixel image.

    Args:
        client: FastAPI test client
        sample_image_small: 1x1 pixel image fixture
    """
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("tiny.png", sample_image_small, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["size"] > 0


def test_upload_large_image(client: TestClient, sample_image_large: BytesIO) -> None:
    """Test uploading a larger image.

    Args:
        client: FastAPI test client
        sample_image_large: 500x500 pixel image fixture
    """
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("large.png", sample_image_large, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["size"] > 0
