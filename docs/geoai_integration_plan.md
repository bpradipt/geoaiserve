# GeoAI Integration Plan

This document outlines the plan to integrate the [geoai](https://github.com/opengeos/geoai) library directly into geoaiserve, replacing the current custom implementations with wrappers around geoai's proven classes.

## Status

- [x] **Phase 1: DINOv3 Integration** - Completed
- [x] **Phase 2: Moondream Integration** - Completed
- [ ] **Phase 3: Sliding Window Endpoints** - Future enhancement

## Current State (After Integration)

The implementation now uses:
- **DINOv3**: geoai's `DINOv3GeoProcessor` with satellite-trained weights (SAT-493M)
- **Moondream**: geoai's `MoondreamGeo` for vision-language inference
- **SAM**: `samgeo` library (already aligned with geoai ecosystem)

### Benefits Achieved

1. **Satellite-optimized weights**: DINOv3 now uses SAT-493M trained weights
2. **Proper normalization**: Uses satellite-specific image normalization stats
3. **Higher resolution features**: 1024-dim features (ViT-L) vs 768-dim (ViT-B)
4. **Consistent with geoai**: Users get expected geoai behavior

## Proposed Architecture

### Hybrid Approach

Use geoai library internally while maintaining our existing API contracts:

```
┌─────────────────────────────────────────────────────────────┐
│                     REST API Endpoints                       │
│  /api/v1/dinov3/*    /api/v1/moondream/*    /api/v1/sam/*   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Service Layer                            │
│  DINOv3Service       MoondreamService       SAMService       │
│  (wrapper)           (wrapper)              (existing)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     geoai Library                            │
│  DINOv3GeoProcessor  MoondreamGeo          SamGeo           │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: DINOv3 Integration

### 1.1 Add geoai Dependency

Update `pyproject.toml`:

```toml
[dependency-groups]
ml = [
    "geoai-py>=0.1.0",  # Add geoai library
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    # ... existing deps
]
```

### 1.2 Update DINOv3Service

Replace current transformers-based implementation with geoai wrapper:

```python
# geoaiserve/models/dinov3_service.py

class DINOv3Service(BaseGeoModel):
    """Service for DINOv3 using geoai's DINOv3GeoProcessor."""

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",  # geoai model name
        device: DeviceType = DeviceType.CPU,
        **kwargs: Any,
    ):
        super().__init__(model_name, device, **kwargs)

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from geoai import DINOv3GeoProcessor

            device = self._get_torch_device()
            self._model = DINOv3GeoProcessor(
                model_name=self.model_name,
                device=device,
            )
            self._loaded = True

        except ImportError as e:
            if self._allow_mock:
                self._model = self._create_mock_model()
                self._is_mock = True
                self._loaded = True
            else:
                raise ImportError(
                    "geoai not installed. Install with: pip install geoai-py"
                ) from e

    def extract_features(
        self,
        image: Image.Image | Path | str,
        return_patch_features: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._loaded:
            self.load()

        if isinstance(image, (Path, str)):
            image = Image.open(image).convert("RGB")

        # Use geoai's extract_features
        features, h_patches, w_patches = self._model.extract_features(image)

        # Convert to our API format
        features_np = features.cpu().numpy()
        cls_token = features_np.mean(axis=(0, 1))  # Global average pooling

        result = {
            "cls_token": cls_token.tolist(),
            "feature_dim": int(features_np.shape[-1]),
            "patch_grid": [h_patches, w_patches],
        }

        if return_patch_features:
            # Reshape to [num_patches, feature_dim]
            patch_features = features_np.reshape(-1, features_np.shape[-1])
            result["patch_features"] = patch_features.tolist()
        else:
            result["patch_features"] = None

        return result

    def compute_patch_similarity(
        self,
        image: Image.Image | Path | str,
        query_points: list[list[float]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._loaded:
            self.load()

        if isinstance(image, (Path, str)):
            image = Image.open(image).convert("RGB")

        # Extract features first
        features, h_patches, w_patches = self._model.extract_features(image)

        # Compute similarity for each query point
        similarity_maps = []
        img_width, img_height = image.size

        for point in query_points:
            x, y = point[0], point[1]
            # Convert pixel coords to patch coords
            patch_x = int((x / img_width) * w_patches)
            patch_y = int((y / img_height) * h_patches)
            patch_x = max(0, min(patch_x, w_patches - 1))
            patch_y = max(0, min(patch_y, h_patches - 1))

            # Use geoai's compute_patch_similarity
            sim_map = self._model.compute_patch_similarity(
                features, patch_x, patch_y
            )
            similarity_maps.append(sim_map.cpu().numpy().tolist())

        return {
            "query_points": query_points,
            "similarity_maps": similarity_maps,
            "map_size": [h_patches, w_patches],
        }
```

### 1.3 Key Differences

| Feature | Current | After geoai Integration |
|---------|---------|------------------------|
| Model | `facebook/dinov2-base` | `dinov3_vitl16` |
| Weights | Generic ImageNet | Satellite-trained (SAT-493M) |
| Normalization | ImageNet stats | Satellite-specific stats |
| Feature dim | 768 | 1024 (ViT-L) |
| Patch size | 14 | 16 |

### 1.4 Config Changes

Update `geoaiserve/config.py`:

```python
# Model names now use geoai naming convention
dinov3_model_name: str = "dinov3_vitl16"  # Options: dinov3_vits16, dinov3_vitb16, dinov3_vitl16
```

### 1.5 Test Updates

Update tests to handle:
- New feature dimensions (1024 for ViT-L)
- New patch grid sizes (depends on input size)
- geoai-specific behavior

---

## Phase 2: Moondream Integration

### 2.1 Update MoondreamService

Replace current implementation with geoai wrapper:

```python
# geoaiserve/models/moondream_service.py

class MoondreamService(BaseGeoModel):
    """Service for Moondream using geoai's MoondreamGeo."""

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        device: DeviceType = DeviceType.CPU,
        **kwargs: Any,
    ):
        super().__init__(model_name, device, **kwargs)
        self.revision = kwargs.get("revision")

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from geoai import MoondreamGeo

            self._model = MoondreamGeo(
                model_name=self.model_name,
                revision=self.revision,
                device=self.device.value,
            )
            self._loaded = True

        except ImportError as e:
            if self._allow_mock:
                self._model = self._create_mock_model()
                self._is_mock = True
                self._loaded = True
            else:
                raise ImportError(
                    "geoai not installed. Install with: pip install geoai-py"
                ) from e

    def caption(
        self,
        image: Image.Image | Path | str,
        length: str = "normal",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._loaded:
            self.load()

        # Use geoai's caption method
        result = self._model.caption(source=image, length=length)

        return {
            "caption": result.get("caption", ""),
            "length": length,
        }

    def query(
        self,
        image: Image.Image | Path | str,
        question: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._loaded:
            self.load()

        result = self._model.query(question=question, source=image)

        return {
            "question": question,
            "answer": result.get("answer", ""),
        }

    def detect(
        self,
        image: Image.Image | Path | str,
        object_type: str,
        return_geodataframe: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._loaded:
            self.load()

        result = self._model.detect(source=image, object_type=object_type)

        return {
            "object_type": object_type,
            "detections": result.get("objects", []),
        }

    def point(
        self,
        image: Image.Image | Path | str,
        object_description: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._loaded:
            self.load()

        result = self._model.point(
            source=image,
            object_description=object_description,
        )

        return {
            "object_description": object_description,
            "points": result.get("points", []),
        }
```

### 2.2 New Features from geoai

Features we gain by using geoai's MoondreamGeo:

1. **Sliding window support** for large images:
   - `detect_sliding_window()`
   - `point_sliding_window()`
   - `query_sliding_window()`
   - `caption_sliding_window()`

2. **Georeferencing**: Auto-transform pixel coords to geographic coords

3. **Vector output**: GeoDataFrame with proper CRS

4. **Moondream3 support**: Reasoning parameter for chain-of-thought

### 2.3 New Endpoints (Optional)

Consider adding endpoints for sliding window processing:

```
POST /api/v1/moondream/detect-tiled
POST /api/v1/moondream/point-tiled
```

---

## Phase 3: Dependency Updates

### 3.1 pyproject.toml Changes

```toml
[dependency-groups]
ml = [
    "geoai-py>=0.1.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    # Remove transformers since geoai handles it
    # "transformers>=4.30.0,<4.50.0",
    "segment-anything-py",
    "einops>=0.8.1",
]
```

### 3.2 Config Updates

```python
# geoaiserve/config.py

# DINOv3 uses geoai model names
dinov3_model_name: str = "dinov3_vitl16"

# Moondream unchanged (geoai uses same naming)
moondream_model_name: str = "vikhyatk/moondream2"
```

---

## Implementation Order

1. **Phase 1a**: Add geoai dependency, update DINOv3Service
2. **Phase 1b**: Update DINOv3 tests, verify all pass
3. **Phase 1c**: Commit DINOv3 changes

4. **Phase 2a**: Update MoondreamService to use geoai
5. **Phase 2b**: Update Moondream tests, verify all pass
6. **Phase 2c**: Commit Moondream changes

7. **Phase 3**: Add sliding window endpoints (optional)

---

## Migration Considerations

### Breaking Changes

1. **Feature dimensions**: DINOv3 will produce 1024-dim features (vs 768)
2. **Patch grid**: Will be different sizes based on model
3. **API response**: Some fields may have different formats

### Backward Compatibility

- Keep existing API contracts unchanged where possible
- Document any breaking changes in API versioning
- Consider v2 endpoints if breaking changes are significant

---

## Testing Strategy

1. **Unit tests**: Update to expect new feature dimensions
2. **Integration tests**: Verify geoai models load correctly
3. **Real model tests**: Run with actual satellite imagery
4. **API contract tests**: Ensure response structure unchanged

---

## Timeline

- Phase 1 (DINOv3): Implement first
- Phase 2 (Moondream): After DINOv3 is stable
- Phase 3 (Sliding window): Future enhancement

---

## References

- [geoai GitHub](https://github.com/opengeos/geoai)
- [geoai DINOv3 Example](https://github.com/opengeos/geoai/blob/main/docs/examples/DINOv3.ipynb)
- [geoai Moondream Example](https://github.com/opengeos/geoai/blob/main/docs/examples/moondream.ipynb)
- [segment-geospatial](https://github.com/opengeos/segment-geospatial)
