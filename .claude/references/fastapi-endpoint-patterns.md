# FastAPI Endpoint Patterns for GeoAI Models

Quick patterns for implementing production-ready FastAPI endpoints.

---

## Model Singleton Pattern

```python
# app/models/manager.py
from geoai.moondream import MoondreamGeo
from geoai.dinov3 import DINOv3GeoProcessor
from samgeo import SamGeo3
import torch

class ModelManager:
    """Singleton pattern for model management"""
    _instances = {}

    @classmethod
    def get_moondream(cls, device: str = "cuda"):
        key = f"moondream_{device}"
        if key not in cls._instances:
            cls._instances[key] = MoondreamGeo(
                model_name="vikhyatk/moondream2",
                device=device
            )
        return cls._instances[key]

    @classmethod
    def get_dinov3(cls):
        if "dinov3" not in cls._instances:
            cls._instances["dinov3"] = DINOv3GeoProcessor()
        return cls._instances["dinov3"]

    @classmethod
    def get_sam(cls, device: str = "cuda"):
        key = f"sam_{device}"
        if key not in cls._instances:
            cls._instances[key] = SamGeo3(
                backend="meta",
                device=device,
                enable_inst_interactivity=True
            )
        return cls._instances[key]

    @classmethod
    def cleanup(cls):
        """Free all models and clear GPU memory"""
        cls._instances.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## File Upload with Validation

```python
# app/utils/upload.py
from fastapi import UploadFile, HTTPException
import tempfile
from pathlib import Path
from typing import Literal
import magic

ALLOWED_MIME_TYPES = {
    "image/tiff",
    "image/jpeg",
    "image/png",
    "image/webp"
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

async def save_upload_file(
    upload_file: UploadFile,
    max_size: int = MAX_FILE_SIZE,
    allowed_types: set = ALLOWED_MIME_TYPES
) -> Path:
    """
    Save uploaded file to temp location with validation.

    Args:
        upload_file: FastAPI UploadFile
        max_size: Maximum file size in bytes
        allowed_types: Set of allowed MIME types

    Returns:
        Path to saved temporary file

    Raises:
        HTTPException: If validation fails
    """
    # Validate content type
    if upload_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    # Read and validate size
    content = await upload_file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {max_size / 1024 / 1024}MB"
        )

    # Verify actual file type (not just extension)
    mime = magic.from_buffer(content, mime=True)
    if mime not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File content doesn't match extension"
        )

    # Save to temp file
    suffix = Path(upload_file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)
```

---

## Request/Response Models

```python
# app/schemas/moondream.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class CaptionRequest(BaseModel):
    length: Literal["short", "normal", "long"] = Field(
        default="normal",
        description="Caption length"
    )

class CaptionResponse(BaseModel):
    caption: str
    length: str
    processing_time: float

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class QueryResponse(BaseModel):
    question: str
    answer: str
    processing_time: float

# app/schemas/sam.py
class SegmentTextRequest(BaseModel):
    prompt: str = Field(..., description="Object to segment")
    min_area: Optional[int] = Field(100, ge=0)
    max_area: Optional[int] = Field(None, ge=0)

class SegmentPointsRequest(BaseModel):
    points: list[list[int]] = Field(..., min_items=1)
    labels: list[int] = Field(..., min_items=1)

    @validator("labels")
    def validate_labels(cls, v):
        if not all(label in [0, 1] for label in v):
            raise ValueError("Labels must be 0 or 1")
        return v

class SegmentBoxesRequest(BaseModel):
    boxes: list[list[int]] = Field(..., min_items=1)
    crs: Optional[str] = Field(None, pattern=r"EPSG:\d+")

# app/schemas/dinov3.py
class SimilarityRequest(BaseModel):
    query_x: int = Field(..., ge=0)
    query_y: int = Field(..., ge=0)

class SimilarityResponse(BaseModel):
    similarity_map: list[list[float]]
    query_coords: tuple[int, int]
    patch_grid: tuple[int, int]
    processing_time: float
```

---

## Error Handling

```python
# app/utils/errors.py
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model errors"""
    pass

class GPUMemoryError(ModelError):
    """GPU out of memory"""
    pass

class InvalidImageError(ModelError):
    """Invalid image format or corrupted"""
    pass

def handle_model_error(e: Exception) -> HTTPException:
    """Convert model errors to HTTP exceptions"""
    logger.error(f"Model error: {e}", exc_info=True)

    if isinstance(e, GPUMemoryError):
        return HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="GPU memory exhausted. Try smaller image or different parameters."
        )

    if isinstance(e, InvalidImageError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    if "CUDA out of memory" in str(e):
        return HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="GPU memory exhausted"
        )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Model processing failed"
    )
```

---

## Endpoint Template

```python
# app/routers/moondream.py
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.models.manager import ModelManager
from app.utils.upload import save_upload_file
from app.utils.errors import handle_model_error
from app.schemas.moondream import CaptionRequest, CaptionResponse
import time
from pathlib import Path

router = APIRouter(prefix="/api/v1/moondream", tags=["moondream"])

@router.post("/caption", response_model=CaptionResponse)
async def caption_image(
    file: UploadFile = File(...),
    length: str = "normal",
    background_tasks: BackgroundTasks = None
):
    """
    Generate natural language caption for an image.

    - **file**: Image file (JPEG, PNG, or GeoTIFF)
    - **length**: Caption length (short, normal, or long)
    """
    temp_path = None
    start_time = time.time()

    try:
        # Save uploaded file
        temp_path = await save_upload_file(file)

        # Get model and process
        model = ModelManager.get_moondream()
        result = model.caption(source=str(temp_path), length=length)

        # Calculate processing time
        processing_time = time.time() - start_time

        return CaptionResponse(
            caption=result["caption"],
            length=result["length"],
            processing_time=processing_time
        )

    except Exception as e:
        raise handle_model_error(e)

    finally:
        # Schedule cleanup
        if temp_path and background_tasks:
            background_tasks.add_task(cleanup_file, temp_path)
        elif temp_path:
            Path(temp_path).unlink(missing_ok=True)

def cleanup_file(path: Path):
    """Background task to clean up temp files"""
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to cleanup {path}: {e}")
```

---

## Async Processing for Heavy Operations

```python
# app/utils/async_processing.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import torch

# Thread pool for CPU-bound tasks
cpu_executor = ThreadPoolExecutor(max_workers=4)

# Thread pool for GPU-bound tasks (limit concurrency)
gpu_executor = ThreadPoolExecutor(max_workers=2)

def run_in_threadpool(executor: ThreadPoolExecutor = cpu_executor):
    """Decorator to run sync functions in thread pool"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                executor,
                lambda: func(*args, **kwargs)
            )
        return wrapper
    return decorator

# Usage example
@run_in_threadpool(gpu_executor)
def process_sam_tiled(image_path: str, prompt: str):
    """Heavy GPU operation - run in dedicated thread pool"""
    sam = ModelManager.get_sam()
    sam.generate_masks_tiled(
        source=image_path,
        prompt=prompt,
        tile_size=1024,
        overlap=128,
        output="result.tif"
    )
    return "result.tif"

@router.post("/segment/tiled")
async def segment_tiled(
    file: UploadFile = File(...),
    prompt: str = "water"
):
    temp_path = await save_upload_file(file)

    # Run heavy processing asynchronously
    result_path = await process_sam_tiled(str(temp_path), prompt)

    return {"status": "success", "result": result_path}
```

---

## Response Streaming for Large Results

```python
from fastapi.responses import StreamingResponse
import io
import numpy as np

@router.post("/segment/stream")
async def segment_stream(file: UploadFile = File(...)):
    """Stream large segmentation results"""
    temp_path = await save_upload_file(file)

    try:
        sam = ModelManager.get_sam()
        sam.set_image(str(temp_path))
        masks = sam.generate_masks(prompt="building")

        # Stream masks as chunked response
        def generate_chunks():
            for i, mask in enumerate(masks):
                # Convert mask to bytes
                buffer = io.BytesIO()
                np.save(buffer, mask)
                buffer.seek(0)

                # Yield chunk with metadata
                yield f"--boundary\r\n"
                yield f"Content-Type: application/octet-stream\r\n"
                yield f"X-Mask-Index: {i}\r\n\r\n"
                yield buffer.read()
                yield f"\r\n"

        return StreamingResponse(
            generate_chunks(),
            media_type="multipart/mixed; boundary=boundary"
        )

    finally:
        Path(temp_path).unlink(missing_ok=True)
```

---

## Caching Results

```python
from functools import lru_cache
from hashlib import sha256
import pickle
from pathlib import Path

CACHE_DIR = Path("/tmp/geoai_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_file_hash(file_path: Path) -> str:
    """Generate hash of file content"""
    hasher = sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def cache_result(func):
    """Cache model results based on file hash and parameters"""
    @wraps(func)
    async def wrapper(file: UploadFile, **kwargs):
        # Save file
        temp_path = await save_upload_file(file)

        # Generate cache key
        file_hash = get_file_hash(temp_path)
        param_hash = sha256(str(kwargs).encode()).hexdigest()
        cache_key = f"{func.__name__}_{file_hash}_{param_hash}.pkl"
        cache_path = CACHE_DIR / cache_key

        # Check cache
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Process and cache
        result = await func(temp_path, **kwargs)

        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

        return result

    return wrapper

@router.post("/caption/cached")
@cache_result
async def caption_cached(file_path: Path, length: str = "normal"):
    model = ModelManager.get_moondream()
    return model.caption(source=str(file_path), length=length)
```

---

## Health Check & Metrics

```python
# app/routers/health.py
from fastapi import APIRouter
import torch
import psutil
from app.models.manager import ModelManager

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
async def health():
    """Basic health check"""
    return {"status": "healthy"}

@router.get("/models")
async def models_status():
    """Check model loading status"""
    return {
        "moondream_loaded": "moondream_cuda" in ModelManager._instances,
        "dinov3_loaded": "dinov3" in ModelManager._instances,
        "sam_loaded": "sam_cuda" in ModelManager._instances,
    }

@router.get("/resources")
async def resources():
    """System resource usage"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
        }

    return {
        **gpu_info,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent
    }

@router.post("/gc")
async def garbage_collect():
    """Trigger garbage collection and GPU cache clear"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "gc_completed"}
```

---

## Startup/Shutdown Events

```python
# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting GeoAI Service...")

    # Preload models (optional - warm start)
    try:
        ModelManager.get_moondream()
        ModelManager.get_dinov3()
        ModelManager.get_sam()
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preloading failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down GeoAI Service...")
    ModelManager.cleanup()
    logger.info("Models cleaned up")

app = FastAPI(
    title="GeoAI Service",
    description="API for geospatial AI models",
    version="1.0.0",
    lifespan=lifespan
)
```

---

## Rate Limiting

```python
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/caption")
@limiter.limit("10/minute")  # 10 requests per minute
async def caption_limited(
    request: Request,
    file: UploadFile = File(...)
):
    # ... implementation
    pass
```

---

## Complete Main Application

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import moondream, dinov3, sam, health
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="GeoAI Service",
    description="Backend API for geospatial AI models",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(moondream.router)
app.include_router(dinov3.router)
app.include_router(sam.router)
app.include_router(health.router)

@app.get("/")
async def root():
    return {
        "service": "GeoAI API",
        "version": "1.0.0",
        "docs": "/docs"
    }
```

---

**Last Updated**: 2025-12-21
**Version**: 1.0
