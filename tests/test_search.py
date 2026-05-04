"""Tests for /api/v1/search template search endpoint."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def test_search_single_target(client: TestClient, sample_image: BytesIO) -> None:
    sample_image.seek(0)
    s = client.post(
        "/api/v1/files/upload",
        files={"file": ("s.png", sample_image, "image/png")},
    )
    assert s.status_code == 200
    sid = s.json()["file_id"]

    sample_image.seek(0)
    t = client.post(
        "/api/v1/files/upload",
        files={"file": ("t.png", sample_image, "image/png")},
    )
    assert t.status_code == 200
    tid = t.json()["file_id"]

    resp = client.post(
        "/api/v1/search/",
        json={
            "source_file_id": sid,
            "target_file_id": tid,
            "top_k_matches_per_target": 3,
            "min_score": 0.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["mode"] == "single"
    assert data["strategy_used"] == "dinov3_multiscale_template_v1"
    assert "matches" in data
    assert isinstance(data["matches"], list)
    assert data["totalMatches"] == len(data["matches"])
    if data["matches"]:
        m0 = data["matches"][0]
        assert "x" in m0 and "y" in m0 and "width" in m0 and "height" in m0
        assert "confidence" in m0


def test_search_multi_target(client: TestClient, sample_image: BytesIO) -> None:
    sample_image.seek(0)
    s = client.post(
        "/api/v1/files/upload",
        files={"file": ("s2.png", sample_image, "image/png")},
    )
    sid = s.json()["file_id"]

    tids = []
    for name in ("a.png", "b.png"):
        sample_image.seek(0)
        r = client.post(
            "/api/v1/files/upload",
            files={"file": (name, sample_image, "image/png")},
        )
        tids.append(r.json()["file_id"])

    resp = client.post(
        "/api/v1/search/",
        json={
            "source_file_id": sid,
            "target_file_ids": tids,
            "top_k_matches_per_target": 2,
            "min_score": 0.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "multiple"
    assert data["results"] is not None
    assert len(data["results"]) == 2
    assert "summary" in data


def test_search_validation_requires_target(client: TestClient, uploaded_file_id: str) -> None:
    r = client.post(
        "/api/v1/search/",
        json={"source_file_id": uploaded_file_id},
    )
    assert r.status_code == 422


def test_search_image_chat_mock(client: TestClient, sample_image: BytesIO) -> None:
    sample_image.seek(0)
    up = client.post(
        "/api/v1/files/upload",
        files={"file": ("chat.png", sample_image, "image/png")},
    )
    assert up.status_code == 200
    fid = up.json()["file_id"]

    resp = client.post(
        "/api/v1/search/",
        json={
            "operation": "image_chat",
            "source_file_id": fid,
            "chat_message": "What do you see?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["clearSelections"] is False
    assert isinstance(data.get("response"), str) and data["response"]
    assert isinstance(data.get("selections"), list) and len(data["selections"]) > 0
    sel0 = data["selections"][0]
    assert "x" in sel0 and "y" in sel0 and "width" in sel0 and "height" in sel0


def test_search_image_chat_requires_message(client: TestClient, uploaded_file_id: str) -> None:
    r = client.post(
        "/api/v1/search/",
        json={
            "operation": "image_chat",
            "source_file_id": uploaded_file_id,
        },
    )
    assert r.status_code == 422


def test_search_image_chat_clear_selections(client: TestClient, uploaded_file_id: str) -> None:
    resp = client.post(
        "/api/v1/search/",
        json={
            "operation": "image_chat",
            "source_file_id": uploaded_file_id,
            "chat_message": "clear all highlights",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["clearSelections"] is True
    assert data["selections"] == []
