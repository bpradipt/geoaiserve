# Problem 2: Cannot Mix File Upload with JSON Request Body

## Issue

When testing DINOv3 and other endpoints that accept both file uploads and JSON parameters, the JSON data is being ignored and default values are used instead.

### Example

```python
response = client.post(
    "/api/v1/dinov3/similarity",
    files={"file": ("test.png", image_bytes, "image/png")},
    json={"query_points": [[50, 50], [75, 75]]},  # This is IGNORED!
)

# Result: query_points = [[100, 100]]  (default value, not [[50, 50], [75, 75]])
```

## Root Cause

**FastAPI limitation**: When using `multipart/form-data` for file uploads, you **cannot** send a JSON request body in the same request. FastAPI will ignore the JSON data.

The current router signature:
```python
async def compute_similarity(
    file: UploadFile = File(..., description="Input image file"),
    request: DINOv3SimilarityRequest = DINOv3SimilarityRequest(query_points=[[100, 100]]),
) -> DINOv3SimilarityResponse:
```

This tries to parse a Pydantic model from JSON, but when `files=` is used in the request, FastAPI uses `Content-Type: multipart/form-data`, not `application/json`.

## Solutions

### Option 1: Individual Form Fields (Recommended for Simple Parameters)

```python
from fastapi import Form

async def compute_similarity(
    file: UploadFile = File(...),
    query_points: str = Form(...),  # JSON string
) -> DINOv3SimilarityResponse:
    # Parse the JSON string
    import json
    points = json.loads(query_points)
```

### Option 2: Separate Endpoints for File Upload vs URL/Base64

```python
# Endpoint 1: Upload file directly
@router.post("/similarity")
async def compute_similarity_upload(
    file: UploadFile = File(...),
    query_points: str = Form(...),
):
    ...

# Endpoint 2: Use URL or base64 with JSON body
@router.post("/similarity/url")
async def compute_similarity_url(
    request: DINOv3SimilarityRequest,
):
    # Fetch image from URL or decode base64
    ...
```

### Option 3: Use Form Data Properly

```python
async def compute_similarity(
    file: UploadFile = File(...),
    query_points: list[list[float]] = Form(...),  # Doesn't work for complex types
):
    ...
```

**Note**: Form() doesn't support complex nested types like `list[list[float]]` directly.

## Impact

Affects all endpoints that:
1. Accept file uploads (`UploadFile`)
2. Have additional Pydantic model parameters

**Affected Endpoints**:
- `/api/v1/dinov3/similarity` ❌
- `/api/v1/dinov3/batch-similarity` ❌
- `/api/v1/sam/generate` ❌
- `/api/v1/sam/predict` ❌
- `/api/v1/sam/batch` ❌
- `/api/v1/moondream/caption` ❌ (when using `data=` in tests)
- `/api/v1/moondream/query` ❌
- All other moondream endpoints ❌

## Recommended Fix

Convert parameters to form fields and parse JSON strings:

```python
async def compute_similarity(
    file: UploadFile = File(...),
    query_points: str = Form(default='[[100, 100]]'),
    model_name: str | None = Form(default=None),
    device: str = Form(default="cpu"),
) -> DINOv3SimilarityResponse:
    import json
    points = json.loads(query_points)
    # ... rest of logic
```

## Alternative: Use ImageInput Schema

The plan already included an `ImageInput` schema that supports URL and base64, which could eliminate the need for file uploads in many cases:

```python
class ImageInput(BaseModel):
    url: HttpUrl | None = None
    base64: str | None = None
```

Then use pure JSON requests without file uploads.

## Test Fix Required

Tests need to send parameters as form data, not JSON:

```python
# WRONG
response = client.post(
    "/api/v1/dinov3/similarity",
    files={"file": file},
    json={"query_points": [[50, 50]]},  # Ignored!
)

# CORRECT
response = client.post(
    "/api/v1/dinov3/similarity",
    files={"file": file},
    data={"query_points": '[[50, 50], [75, 75]]'},  # JSON string in form data
)
```

## Lesson Learned

**Never mix `files=` with `json=` in FastAPI requests**. Use form data for all parameters when uploading files, or use separate endpoints for file upload vs JSON-only requests.

## Next Steps

1. Decide on approach (Form fields vs separate endpoints)
2. Update all affected routers
3. Update tests to use form data
4. Document API behavior in OpenAPI docs

## Files to Update

- `geoaiserve/routers/sam.py`
- `geoaiserve/routers/moondream.py`
- `geoaiserve/routers/dinov3.py`
- All test files in `tests/`
