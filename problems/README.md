# Testing Phase 2 - Problems Discovered and Documented

This directory contains detailed documentation of problems discovered during Phase 2 testing.

## Summary

**Test Coverage**: 36 tests across 3 model services
**Passing Tests**: 19/36 (53%)
**Failing Tests**: 17/36 (47%)

## Problems Identified

### ✅ Problem 1: TestClient Lifespan Events - **FIXED**

**File**: `problem1_testclient_lifespan.md`

- **Severity**: High (blocked all model endpoint tests)
- **Status**: ✅ Fixed
- **Impact**: Models weren't being registered during tests
- **Solution**: Use `with TestClient(app) as client:` in conftest.py

---

### ⚠️ Problem 2: Multipart/Form-Data vs JSON Conflict - **REQUIRES DECISION**

**File**: `problem2_multipart_json_conflict.md`

- **Severity**: Critical (affects 13 of 13 model endpoints)
- **Status**: ⚠️ Documented, needs architectural decision
- **Impact**: All endpoints that accept file uploads ignore JSON parameters
- **Root Cause**: FastAPI cannot parse JSON body when using multipart/form-data

**Affected Endpoints**:
```
- /api/v1/sam/generate
- /api/v1/sam/predict
- /api/v1/sam/batch
- /api/v1/moondream/caption
- /api/v1/moondream/query
- /api/v1/moondream/detect
- /api/v1/moondream/point
- /api/v1/dinov3/features
- /api/v1/dinov3/similarity
- /api/v1/dinov3/batch-similarity
```

**Possible Solutions**:
1. **Use Form fields** - Convert all parameters to Form() fields (simple but verbose)
2. **Separate endpoints** - One for file upload, one for URL/base64 (cleaner API)
3. **JSON in form field** - Send JSON as string in form data (hacky)
4. **URL/base64 only** - Use ImageInput schema, no file uploads (elegant but limits use cases)

**Recommendation**: Option 2 (Separate endpoints) provides the best API design:
- `/api/v1/{model}/task` - Accept JSON with ImageInput (url/base64)
- `/api/v1/{model}/task/upload` - Accept file upload with form parameters

---

## Test Results Breakdown

### ✅ Common Endpoints: 7/7 (100%)
- Health check ✓
- List models ✓
- Get model info (SAM, Moondream, DINOv3) ✓
- Invalid model (404 handling) ✓
- OpenAPI schema ✓

### ⚠️ DINOv3 Endpoints: 3/11 (27%)
- Feature extraction (basic) ✓
- Feature extraction (with/without patches) ✓
- Similarity (default params) ❌ (param parsing)
- Similarity (multiple points) ❌ (param parsing)
- Batch similarity ❌ (param parsing)

### ⚠️ Moondream Endpoints: 4/10 (40%)
- Caption (default) ✓
- Caption (length variations) ❌ (param parsing)
- Query (default) ❌ (param parsing)
- Query (variations) Mixed ✓/❌
- Detect ✓
- Point ❌ (param parsing)

### ❌ SAM Endpoints: 0/8 (0%)
- All failing due to param parsing issues ❌

---

## Next Steps

### Immediate Actions Required:

1. **Architectural Decision**: Choose approach for handling file uploads + parameters
2. **Router Refactoring**: Update all 13 endpoints based on chosen approach
3. **Test Updates**: Modify tests to match new API design
4. **Documentation**: Update OpenAPI docs to reflect correct usage

### Alternative Path:

Consider implementing Problem 2 solutions in parallel:
- Keep current file upload endpoints for backward compatibility
- Add new `/upload` endpoints with proper form handling
- Add JSON-only endpoints using ImageInput schema

This provides maximum flexibility while maintaining clean API design.

---

## Lessons Learned

1. **Always write tests first** - We would have caught these issues during development
2. **Lifespan events matter** - TestClient needs context manager for proper initialization
3. **FastAPI limitations** - Cannot mix multipart file uploads with JSON bodies
4. **API design is critical** - File upload pattern needs careful consideration upfront

---

## Test Execution

Run tests:
```bash
uv run pytest -v
```

Run specific test file:
```bash
uv run pytest tests/test_common.py -v
```

Run with coverage:
```bash
uv run pytest --cov=geoaiserve --cov-report=html
```

---

## Impact Assessment

**Phase 2 Status**: ⚠️ **Implementation complete, but API design needs revision**

The core model services work correctly (as evidenced by the 19 passing tests), but the HTTP API layer has a fundamental design flaw that prevents proper parameter passing when uploading files.

This must be addressed before Phase 2 can be considered production-ready.
