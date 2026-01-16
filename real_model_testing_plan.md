# Real Model Testing with GeoTIFF Support

## Summary

Add real model testing infrastructure to geoaiserve with:
- Optional ML dependencies (torch, transformers, samgeo)
- GeoTIFF loading utility for converting to RGB
- Pytest markers to distinguish mock vs real tests
- Real model test files for SAM, Moondream, DINOv3
- User-provided GeoTIFF files in tests/data/

**Scope**: Basic inference verification on CPU

---

## Current State

- 102 mock tests passing
- Models fall back to mocks when ML deps missing
- `rasterio` already installed for GeoTIFF
- `validate_geotiff()` exists in file_handler.py but unused
- All models convert to RGB via PIL

---

## Implementation Phases

### Phase 1: Add Optional ML Dependencies
**Commit**: `feat(deps): add optional ML dependencies for real model testing`

**File**: `pyproject.toml`

Add dependency groups:
```toml
[dependency-groups]
ml = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "transformers>=4.30.0",
    "segment-anything-py",
]

dev-ml = [
    {include-group = "dev"},
    {include-group = "ml"},
]
```

Add pytest markers:
```toml
[tool.pytest.ini_options]
markers = [
    "mock: tests using mock models (default)",
    "real_model: tests requiring real ML models",
    "real_sam: SAM-specific real tests",
    "real_moondream: Moondream-specific real tests",
    "real_dinov3: DINOv3-specific real tests",
    "geotiff: tests using GeoTIFF files",
    "slow: long-running tests",
]
```

---

### Phase 2: Create GeoTIFF Loading Utility
**Commit**: `feat(utils): add GeoTIFF loading utility with RGB conversion`

**New File**: `geoaiserve/utils/geotiff_loader.py`

Key components:
- `GeoTIFFData` dataclass with image, metadata, original_shape, dtype
- `load_geotiff_as_rgb(file_path, bands, normalize)` - Load GeoTIFF, convert to RGB PIL Image
- `save_rgb_as_geotiff(image, output_path, reference_metadata)` - Save with georeferencing
- `_normalize_to_uint8(array)` - Helper for value normalization

**Update**: `geoaiserve/utils/__init__.py` - Export new functions

---

### Phase 3: Add Pytest Markers and Fixtures
**Commit**: `feat(tests): add pytest markers and fixtures for real model testing`

**New File**: `tests/markers.py`
```python
ML_AVAILABLE = check if torch/transformers installed
SAMGEO_AVAILABLE = check if samgeo installed

skip_without_ml = pytest.mark.skipif(not ML_AVAILABLE, ...)
skip_without_samgeo = pytest.mark.skipif(not SAMGEO_AVAILABLE, ...)
```

**Update**: `tests/conftest.py`

New fixtures:
| Fixture | Scope | Purpose |
|---------|-------|---------|
| `test_data_dir` | function | Path to tests/data/ |
| `geotiff_paths` | function | List of .tif/.tiff files |
| `sample_geotiff` | function | First available GeoTIFF |
| `uploaded_geotiff_id` | function | Upload GeoTIFF, return file_id |
| `real_sam_service` | session | Real SAM model |
| `real_moondream_service` | session | Real Moondream model |
| `real_dinov3_service` | session | Real DINOv3 model |

---

### Phase 4: Set Up Test Data Directory
**Commit**: `docs(tests): add GeoTIFF test data directory and documentation`

**Create**: `tests/data/.gitkeep`
**Create**: `tests/data/README.md` - Document test data requirements

**Update**: `.gitignore`
```gitignore
tests/data/*.tif
tests/data/*.tiff
!tests/data/.gitkeep
!tests/data/README.md
```

---

### Phase 5: Create Real Model Tests
**Commit**: `feat(tests): add real model test files for SAM, Moondream, and DINOv3`

**New Files**:

| File | Test Classes | Key Tests |
|------|--------------|-----------|
| `tests/test_real_sam.py` | TestRealSAMService | Load verification, predict with GeoTIFF, generate masks |
| `tests/test_real_moondream.py` | TestRealMoondreamService | Load verification, caption, query, detect |
| `tests/test_real_dinov3.py` | TestRealDINOv3Service, TestDINOv3FeatureConsistency | Features, similarity, determinism |
| `tests/test_geotiff_loader.py` | TestGeoTIFFLoader, TestGeoTIFFSaveRoundtrip | Unit tests for loader utility |

All test classes marked with `@pytest.mark.real_model` and `@pytest.mark.slow`.

---

### Phase 6: Mark Existing Tests
**Commit**: `refactor(tests): mark existing tests with mock marker for filtering`

**Update**: `tests/test_sam.py`, `tests/test_moondream.py`, `tests/test_dinov3.py`

Add at top of each file:
```python
pytestmark = pytest.mark.mock
```

---

### Phase 7: Documentation
**Commit**: `docs: add comprehensive testing documentation`

**Update**: `README.md` - Add testing section with markers table
**Update**: `Makefile` - Add real model test targets
**Create**: `TESTING.md` - Detailed testing guide

---

## Files Summary

### Create (8 files)
| File | Purpose |
|------|---------|
| `geoaiserve/utils/geotiff_loader.py` | GeoTIFF to RGB conversion |
| `tests/markers.py` | Skip decorators for dependencies |
| `tests/test_real_sam.py` | Real SAM model tests |
| `tests/test_real_moondream.py` | Real Moondream tests |
| `tests/test_real_dinov3.py` | Real DINOv3 tests |
| `tests/test_geotiff_loader.py` | GeoTIFF loader unit tests |
| `tests/data/.gitkeep` | Track directory |
| `tests/data/README.md` | Test data documentation |

### Modify (8 files)
| File | Changes |
|------|---------|
| `pyproject.toml` | ML deps, pytest markers |
| `geoaiserve/utils/__init__.py` | Export loader |
| `tests/conftest.py` | GeoTIFF and real model fixtures |
| `tests/test_sam.py` | Add mock marker |
| `tests/test_moondream.py` | Add mock marker |
| `tests/test_dinov3.py` | Add mock marker |
| `.gitignore` | Ignore test GeoTIFFs |
| `README.md` | Testing docs |
| `Makefile` | Test targets |

---

## Usage

### Installation
```bash
# Default (mock tests only)
uv sync

# With ML dependencies
uv sync --group ml
```

### Running Tests
```bash
# All tests
pytest

# Only mock tests (fast)
pytest -m mock

# Only real model tests
pytest -m real_model

# Specific model
pytest -m real_dinov3

# Skip slow tests
pytest -m "not slow"
```

### Test Data Setup
```bash
# Copy your GeoTIFF files
cp /path/to/satellite.tif tests/data/
```

---

## Verification

After implementation:

1. **Mock tests still work without ML deps**:
   ```bash
   uv sync
   pytest -m mock -v
   ```

2. **Real tests skip gracefully without deps**:
   ```bash
   pytest -m real_model --collect-only
   # Should show "skipped" for each test
   ```

3. **With ML deps and test data**:
   ```bash
   uv sync --group ml
   cp /path/to/test.tif tests/data/
   pytest -m real_model -v
   ```

4. **GeoTIFF loader works**:
   ```bash
   python -c "from geoaiserve.utils.geotiff_loader import load_geotiff_as_rgb; print('OK')"
   ```

---

## Test Count Estimate

| Category | Tests |
|----------|-------|
| Existing mock tests | 102 |
| New GeoTIFF loader tests | ~8 |
| New real SAM tests | ~4 |
| New real Moondream tests | ~5 |
| New real DINOv3 tests | ~8 |
| **Total** | **~127 tests** |
