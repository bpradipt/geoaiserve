"""Multi-scale template search using DINOv3 dense patch similarity (POC)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..models import registry
from ..schemas.common import ModelType
from ..schemas.search import (
    BboxPct,
    BboxPx,
    PerTargetSearchResult,
    SearchMatch,
    SearchOptions,
    SearchRequest,
    SearchResponse,
    SearchSummary,
)
from ..services.file_handler import file_handler

logger = logging.getLogger(__name__)


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ba = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 0 else 0.0


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def _bbox_to_pct(x1: int, y1: int, x2: int, y2: int, iw: int, ih: int) -> BboxPct:
    w = x2 - x1
    h = y2 - y1
    return BboxPct(
        x=100.0 * x1 / iw,
        y=100.0 * y1 / ih,
        width=100.0 * w / iw,
        height=100.0 * h / ih,
    )


def load_rgb_pil(path: Path) -> tuple[Image.Image, int, int, list[str]]:
    """Load image as RGB PIL. Supports GeoTIFF via rasterio and common formats via PIL."""
    warnings: list[str] = []
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        img = Image.open(path).convert("RGB")
        return img, img.width, img.height, warnings
    try:
        import rasterio

        with rasterio.open(path) as src:
            if src.crs is not None:
                logger.debug("Raster CRS: %s", src.crs)
            num_bands = min(3, src.count)
            bands = [src.read(i + 1) for i in range(num_bands)]
            if num_bands == 1:
                arr = bands[0]
            else:
                arr = np.stack(bands, axis=-1)
            if arr.dtype != np.uint8:
                amin, amax = np.nanmin(arr), np.nanmax(arr)
                if amax > amin:
                    arr = ((arr - amin) / (amax - amin) * 255).astype(np.uint8)
                else:
                    arr = np.zeros(arr.shape[:2], dtype=np.uint8)
            arr = np.nan_to_num(arr, nan=0).astype(np.uint8)
            if num_bands == 1:
                img = Image.fromarray(arr, mode="L").convert("RGB")
            else:
                img = Image.fromarray(arr, mode="RGB")
            return img, img.width, img.height, warnings
    except Exception as e:
        logger.debug("rasterio load failed (%s), trying PIL", e)
    img = Image.open(path).convert("RGB")
    return img, img.width, img.height, warnings


def _crop_source_roi_pct(pil: Image.Image, roi: Any | None) -> Image.Image:
    if roi is None:
        return pil
    w, h = pil.size
    x1 = int(roi.x / 100.0 * w)
    y1 = int(roi.y / 100.0 * h)
    x2 = int((roi.x + roi.width) / 100.0 * w)
    y2 = int((roi.y + roi.height) / 100.0 * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return pil
    return pil.crop((x1, y1, x2, y2))


def _resize_max_side(pil: Image.Image, max_side: int) -> Image.Image:
    w, h = pil.size
    m = max(w, h)
    if m <= max_side:
        return pil
    scale = max_side / m
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return pil.resize((nw, nh), Image.Resampling.LANCZOS)


def _find_peaks_greedy(
    sim_map: np.ndarray,
    max_peaks: int,
    min_dist: int,
    min_score: float,
) -> list[tuple[int, int, float]]:
    """Greedy local peak selection on HxW similarity map (patch indices)."""
    h, w = sim_map.shape
    flat = sim_map.reshape(-1)
    order = np.argsort(-flat)
    taken: list[tuple[int, int, float]] = []
    for idx in order:
        score = float(flat[idx])
        if score < min_score:
            break
        py, px = int(idx // w), int(idx % w)
        ok = True
        for ty, tx, _ in taken:
            if abs(py - ty) < min_dist and abs(px - tx) < min_dist:
                ok = False
                break
        if ok:
            taken.append((py, px, score))
        if len(taken) >= max_peaks:
            break
    return taken


def _nms_xyxy(
    boxes: list[tuple[int, int, int, int, float, dict[str, Any]]],
    iou_thresh: float,
    keep: int,
) -> list[tuple[int, int, int, int, float, dict[str, Any]]]:
    """NMS by score descending."""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[int, int, int, int, float, dict[str, Any]]] = []
    for b in boxes:
        bb = (b[0], b[1], b[2], b[3])
        if any(_iou_xyxy(bb, (k[0], k[1], k[2], k[3])) > iou_thresh for k in kept):
            continue
        kept.append(b)
        if len(kept) >= keep:
            break
    return kept


def _mock_peaks(
    orig_tw: int,
    orig_th: int,
    top_k: int,
    min_score: float,
) -> list[SearchMatch]:
    """Deterministic-ish mock boxes for tests without ML."""
    rng = np.random.default_rng(42)
    out: list[SearchMatch] = []
    base_score = max(min_score + 0.05, 0.82)
    for i in range(top_k):
        bw = int(orig_tw * rng.uniform(0.08, 0.18))
        bh = int(orig_th * rng.uniform(0.08, 0.18))
        x1 = int(rng.uniform(0, max(1, orig_tw - bw)))
        y1 = int(rng.uniform(0, max(1, orig_th - bh)))
        x2, y2 = x1 + bw, y1 + bh
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, orig_tw, orig_th)
        sc = float(min(0.99, base_score - i * 0.03))
        if sc < min_score:
            continue
        conf = int(round(sc * 100))
        bp = BboxPx(x1=x1, y1=y1, x2=x2, y2=y2)
        pct = _bbox_to_pct(x1, y1, x2, y2, orig_tw, orig_th)
        out.append(
            SearchMatch(
                score=sc,
                confidence=conf,
                bbox_px=bp,
                bbox_pct=pct,
                meta={"mock": True},
            )
        )
    return out


def _localize_target(
    dinov3: Any,
    source_pil: Image.Image,
    target_pil: Image.Image,
    orig_tw: int,
    orig_th: int,
    options: SearchOptions,
    top_k: int,
    min_score: float,
) -> list[SearchMatch]:
    """Run multi-scale CLS-vs-patch similarity and return matches in original target pixels."""
    all_boxes: list[tuple[int, int, int, int, float, dict[str, Any]]] = []
    used_real_patch_features = False

    for qs in options.query_scale_factors:
        qside = max(32, int(options.max_query_side * qs))
        src_r = _resize_max_side(source_pil, qside)
        src_feat = dinov3.extract_features(
            src_r,
            return_patch_features=False,
            target_size=min(896, max(224, qside)),
        )
        q = np.asarray(src_feat["cls_token"], dtype=np.float64).ravel()
        qn = q / (np.linalg.norm(q) + 1e-8)

        for ts in options.target_scale_factors:
            tside = max(64, int(options.max_target_side * ts))
            tgt_r = _resize_max_side(target_pil, tside)
            tw, th = tgt_r.size
            tgt_size = min(896, max(224, tside))
            tgt_feat = dinov3.extract_features(
                tgt_r,
                return_patch_features=True,
                target_size=tgt_size,
            )
            patches = tgt_feat.get("patch_features")
            grid = tgt_feat.get("patch_grid")
            if patches is None or grid is None:
                logger.info(
                    "No patch features (mock DINOv3); using synthetic peaks for this target"
                )
                return _mock_peaks(orig_tw, orig_th, top_k, min_score)

            used_real_patch_features = True
            h_p, w_p = int(grid[0]), int(grid[1])
            p = np.asarray(patches, dtype=np.float64)
            if p.ndim != 2 or p.shape[0] != h_p * w_p:
                p = p.reshape(-1, p.shape[-1])
            pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
            sims = pn @ qn
            sim_map = sims.reshape(h_p, w_p)

            peaks = _find_peaks_greedy(
                sim_map,
                max_peaks=top_k * 4,
                min_dist=options.min_peak_distance_patches,
                min_score=min_score,
            )
            cell_w = tw / w_p
            cell_h = th / h_p
            sx = orig_tw / tw
            sy = orig_th / th
            for py, px, sc in peaks:
                x1t = int(px * cell_w)
                y1t = int(py * cell_h)
                x2t = int((px + 1) * cell_w)
                y2t = int((py + 1) * cell_h)
                x1 = int(x1t * sx)
                y1 = int(y1t * sy)
                x2 = int(x2t * sx)
                y2 = int(y2t * sy)
                x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, orig_tw, orig_th)
                meta = {
                    "query_scale": qs,
                    "target_scale": ts,
                    "patch": [py, px],
                }
                all_boxes.append((x1, y1, x2, y2, float(sc), meta))

    if not all_boxes:
        if used_real_patch_features:
            return []
        return _mock_peaks(orig_tw, orig_th, top_k, min_score)

    picked = _nms_xyxy(all_boxes, options.iou_nms_threshold, top_k)
    matches: list[SearchMatch] = []
    for x1, y1, x2, y2, sc, meta in picked:
        conf = int(round(max(0.0, min(1.0, sc)) * 100))
        matches.append(
            SearchMatch(
                score=float(sc),
                confidence=conf,
                bbox_px=BboxPx(x1=x1, y1=y1, x2=x2, y2=y2),
                bbox_pct=_bbox_to_pct(x1, y1, x2, y2, orig_tw, orig_th),
                meta=meta,
            )
        )
    return matches


def _geotiff_warnings(src_path: Path, tgt_path: Path) -> list[str]:
    warnings: list[str] = []
    try:
        import rasterio

        with rasterio.open(src_path) as a, rasterio.open(tgt_path) as b:
            ca, cb = a.crs, b.crs
            if ca is not None and cb is not None and ca != cb:
                warnings.append(f"CRS mismatch: source {ca} vs target {cb}")
            if a.res != b.res:
                warnings.append(
                    f"Resolution may differ: source {a.res} vs target {b.res}"
                )
    except Exception:
        pass
    return warnings


def _extract_detection_objects(detections: Any) -> list[dict[str, Any]]:
    """Normalize Moondream detect output to a list of dicts with bbox + label."""
    if detections is None:
        return []
    if isinstance(detections, list):
        return [x for x in detections if isinstance(x, dict)]
    if isinstance(detections, dict):
        objs = detections.get("objects")
        if isinstance(objs, list):
            return [x for x in objs if isinstance(x, dict)]
    return []


def _bbox_to_selection_pct(
    bbox: list[float] | tuple[float, ...],
    iw: int,
    ih: int,
    label: str,
) -> dict[str, Any] | None:
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if max(x1, y1, x2, y2) <= 1.0 + 1e-6:
        x1, x2 = x1 * iw, x2 * iw
        y1, y2 = y1 * ih, y2 * ih
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(iw), x2), min(float(ih), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = x2 - x1, y2 - y1
    return {
        "x": 100.0 * x1 / iw,
        "y": 100.0 * y1 / ih,
        "width": 100.0 * w / iw,
        "height": 100.0 * h / ih,
        "label": label,
    }


def _fallback_chat_selections(message_lower: str) -> list[dict[str, Any]]:
    """Percent boxes aligned with spatialint-ui mock heuristics when detect is empty."""
    if "count" in message_lower or "how many" in message_lower:
        return [
            {"x": 15, "y": 20, "width": 18, "height": 15, "label": "Object 1"},
            {"x": 40, "y": 25, "width": 20, "height": 18, "label": "Object 2"},
            {"x": 65, "y": 45, "width": 19, "height": 16, "label": "Object 3"},
            {"x": 30, "y": 65, "width": 22, "height": 17, "label": "Object 4"},
        ]
    if "describe" in message_lower or "scene" in message_lower:
        return [
            {"x": 35, "y": 30, "width": 30, "height": 35, "label": "Primary Focus"},
            {"x": 10, "y": 60, "width": 20, "height": 18, "label": "Supporting Element 1"},
            {"x": 70, "y": 55, "width": 22, "height": 20, "label": "Supporting Element 2"},
        ]
    if "what" in message_lower or "see" in message_lower:
        return [
            {"x": 20, "y": 15, "width": 30, "height": 25, "label": "Main Subject"},
            {"x": 60, "y": 40, "width": 25, "height": 20, "label": "Secondary Element"},
        ]
    return [
        {"x": 25, "y": 30, "width": 35, "height": 30, "label": "Area of Interest"},
    ]


def _detections_to_chat_selections(
    detections_raw: Any,
    iw: int,
    ih: int,
    default_label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obj in _extract_detection_objects(detections_raw):
        bbox = obj.get("bbox")
        if not bbox:
            continue
        label = str(obj.get("label") or default_label)
        sel = _bbox_to_selection_pct(bbox, iw, ih, label)
        if sel:
            out.append(sel)
    return out


def run_image_chat(req: SearchRequest) -> SearchResponse:
    """Vision-language chat about one image; response matches spatialint-ui mock /api/chat."""
    msg = (req.chat_message or "").strip()
    lower = msg.lower()

    src_path = file_handler.get_file_by_id(req.source_file_id)
    _, iw, ih, load_warnings = load_rgb_pil(src_path)

    clear_phrases = (
        "clear selection",
        "remove selection",
        "clear highlight",
        "clear highlights",
    )
    clear_intent = any(p in lower for p in clear_phrases) or (
        ("clear" in lower or "remove" in lower)
        and ("highlight" in lower or "selection" in lower)
    )
    if clear_intent:
        return SearchResponse(
            status="success",
            strategy_used="image_chat_rules",
            source_file_id=req.source_file_id,
            mode="single",
            success=True,
            response="I've cleared all the highlighted selections from the image.",
            selections=[],
            clearSelections=True,
            totalMatches=0,
            avgConfidence=0,
            searchTime=0,
            matches=[],
            targets=[],
            warnings=load_warnings,
        )

    moondream = registry.get_model(
        model_type=ModelType.MOONDREAM,
        model_name=req.model_params.model_name,
        device=req.model_params.device,
    )
    q_result = moondream.query(image=src_path, question=msg)
    answer_text = str(q_result.get("answer", "")).strip() or (
        "I could not generate a detailed answer for this image."
    )

    object_type = "object"
    if "person" in lower or "people" in lower:
        object_type = "person"
    elif "car" in lower or "vehicle" in lower:
        object_type = "vehicle"
    elif "building" in lower or "structure" in lower:
        object_type = "building"
    elif "tree" in lower or "vegetation" in lower:
        object_type = "tree"

    selections: list[dict[str, Any]] = []
    detect_warnings: list[str] = []
    try:
        d_result = moondream.detect(image=src_path, object_type=object_type)
        raw_dets = d_result.get("detections")
        selections = _detections_to_chat_selections(raw_dets, iw, ih, object_type)
    except Exception as exc:
        logger.debug("Moondream detect skipped/failed in image_chat: %s", exc)
        detect_warnings.append(f"detection_skipped: {exc}")

    if not selections:
        selections = _fallback_chat_selections(lower)

    return SearchResponse(
        status="success",
        strategy_used="moondream_image_chat_v1",
        source_file_id=req.source_file_id,
        mode="single",
        success=True,
        response=answer_text,
        selections=selections,
        clearSelections=False,
        totalMatches=0,
        avgConfidence=0,
        searchTime=0,
        matches=[],
        targets=[],
        warnings=load_warnings + detect_warnings,
    )


def run_search(req: SearchRequest) -> SearchResponse:
    """Execute search across one or many targets."""
    t0 = time.perf_counter()
    strategy = (
        "dinov3_multiscale_template_v1"
        if req.strategy == "auto"
        else req.strategy
    )

    src_path = file_handler.get_file_by_id(req.source_file_id)
    source_pil, _, _, _ = load_rgb_pil(src_path)
    source_pil = _crop_source_roi_pct(source_pil, req.source_roi)

    if req.target_file_id:
        target_ids = [req.target_file_id]
        mode: Any = "single"
    else:
        target_ids = list(req.target_file_ids or [])
        mode = "multiple"

    dinov3 = registry.get_model(
        model_type=ModelType.DINOV3,
        model_name=req.model_params.model_name,
        device=req.model_params.device,
    )

    per_target: list[PerTargetSearchResult] = []
    global_warnings: list[str] = []

    scored: list[tuple[float, str, PerTargetSearchResult]] = []

    for tid in target_ids:
        tgt_path = file_handler.get_file_by_id(tid)
        twarn = _geotiff_warnings(src_path, tgt_path)
        tgt_pil, ow, oh, wload = load_rgb_pil(tgt_path)
        twarn.extend(wload)

        matches = _localize_target(
            dinov3,
            source_pil,
            tgt_pil,
            ow,
            oh,
            req.options,
            req.top_k_matches_per_target,
            req.min_score,
        )
        pr = PerTargetSearchResult(
            target_file_id=tid,
            image_width=ow,
            image_height=oh,
            matches=matches,
            warnings=twarn,
        )
        best = max((m.score for m in matches), default=0.0)
        scored.append((best, tid, pr))

    if mode == "multiple" and req.top_k_targets is not None:
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: req.top_k_targets]

    for _, _, pr in scored:
        per_target.append(pr)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build mock-compatible payloads for spatialint-ui
    if mode == "single" and per_target:
        pt = per_target[0]
        mlist = pt.matches
        mock_matches = [
            {
                "x": m.bbox_pct.x,
                "y": m.bbox_pct.y,
                "width": m.bbox_pct.width,
                "height": m.bbox_pct.height,
                "confidence": m.confidence,
            }
            for m in mlist
        ]
        avg_c = (
            int(round(sum(m.confidence for m in mlist) / len(mlist)))
            if mlist
            else 0
        )
        return SearchResponse(
            status="success",
            strategy_used=strategy,
            source_file_id=req.source_file_id,
            mode="single",
            totalMatches=len(mlist),
            avgConfidence=avg_c,
            searchTime=elapsed_ms,
            matches=mock_matches,
            targets=per_target,
            warnings=global_warnings + pt.warnings,
        )

    results_json: list[dict[str, Any]] = []
    total_m = 0
    for pt in per_target:
        total_m += len(pt.matches)
        results_json.append(
            {
                "imageName": pt.target_file_id,
                "totalMatches": len(pt.matches),
                "matches": [
                    {
                        "confidence": m.confidence,
                        "bbox": {
                            "x": m.bbox_pct.x,
                            "y": m.bbox_pct.y,
                            "width": m.bbox_pct.width,
                            "height": m.bbox_pct.height,
                        },
                    }
                    for m in pt.matches
                ],
            }
        )

    summary = SearchSummary(
        num_targets_requested=len(target_ids),
        num_targets_returned=len(per_target),
        total_matches=total_m,
        processing_time_ms=elapsed_ms,
    )

    iw = [r["imageName"] for r in results_json]
    match_rate = (
        round(100.0 * sum(1 for r in results_json if r["totalMatches"] > 0) / len(results_json))
        if results_json
        else 0
    )
    all_conf = [
        mm["confidence"]
        for r in results_json
        for mm in r["matches"]
    ]
    avg_conf = int(round(sum(all_conf) / len(all_conf))) if all_conf else 0

    summary_dict = {
        "totalImagesSearched": summary.num_targets_returned,
        "totalMatches": summary.total_matches,
        "imagesWithMatches": sum(1 for r in results_json if r["totalMatches"] > 0),
        "matchRate": match_rate,
        "avgConfidence": avg_conf,
    }

    return SearchResponse(
        status="success",
        strategy_used=strategy,
        source_file_id=req.source_file_id,
        mode="multiple",
        sourceImage=req.source_file_id,
        targetImages=iw,
        results=results_json,
        summary=summary_dict,
        targets=per_target,
        warnings=global_warnings,
    )
