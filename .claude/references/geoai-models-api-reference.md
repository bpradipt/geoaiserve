# GeoAI Models API Reference

Quick reference for implementing FastAPI endpoints that wrap moondream, DINOv3, and SamGeo models from the opengeos/geoai toolkit.

**Sources:**
- https://github.com/opengeos/geoai
- https://github.com/opengeos/segment-geospatial

---

## 📖 Table of Contents

1. [Moondream - Vision Language Model](#moondream---vision-language-model)
2. [DINOv3 - Feature Extraction & Similarity](#dinov3---feature-extraction--similarity)
3. [SamGeo3 - Segmentation Anything Model](#samgeo3---segmentation-anything-model)
4. [FastAPI Integration Patterns](#fastapi-integration-patterns)
5. [Common Patterns](#common-patterns)

---

## Moondream - Vision Language Model

**Purpose**: Vision-language model for image captioning, VQA, object detection, and point localization

### Initialization

```python
from geoai.moondream import MoondreamGeo

processor = MoondreamGeo(
    model_name="vikhyatk/moondream2",  # or "vikhyatk/moondream3-preview"
    revision="2025-06-21",              # Optional checkpoint
    device="cuda",                       # or "cpu"
    compile_model=False                  # Optional model compilation
)
```

### 1. Image Captioning

Generate natural language descriptions of images.

```python
# Basic usage
result = processor.caption(
    source="path/to/image.tif",  # or PIL Image, numpy array
    length="normal"               # "short", "normal", or "long"
)

# Output
{
    "caption": "An aerial view of a residential neighborhood...",
    "length": "normal"
}
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/moondream/caption
Request: { "image": <file>, "length": "normal" }
Response: { "caption": "...", "length": "normal" }
```

### 2. Visual Question Answering (VQA)

Ask questions about images.

```python
result = processor.query(
    question="How many buildings are in the image?",
    source="path/to/image.tif"
)

# Output
{
    "question": "How many buildings are in the image?",
    "answer": "There are approximately 15 buildings visible..."
}
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/moondream/query
Request: { "image": <file>, "question": "..." }
Response: { "question": "...", "answer": "..." }
```

### 3. Object Detection

Detect and locate objects in images with georeferencing.

```python
result = processor.detect(
    source="path/to/image.tif",
    object_type="building",
    output_path="buildings.geojson"  # Optional
)

# Returns GeoDataFrame with bounding boxes
# If GeoTIFF input: georeferenced coordinates
# If regular image: pixel coordinates
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/moondream/detect
Request: { "image": <file>, "object_type": "building" }
Response: { "detections": [...], "count": 15, "format": "geojson" }
```

### 4. Point Localization

Find centroids of specific objects.

```python
result = processor.point(
    source="path/to/image.tif",
    object_description="building",
    output_path="centroids.geojson"  # Optional
)

# Returns GeoDataFrame with point geometries
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/moondream/point
Request: { "image": <file>, "object": "building" }
Response: { "points": [...], "count": 15, "format": "geojson" }
```

### Image Input Formats

Moondream accepts:
- **File paths**: `"path/to/image.tif"`
- **PIL Images**: `Image.open("image.jpg")`
- **NumPy arrays**: `np.array(...)`
- **GeoTIFF** (automatically preserves georeferencing)

### Convenience Functions

```python
from geoai.moondream import (
    moondream_caption,
    moondream_query,
    moondream_detect,
    moondream_point
)

# Use without creating processor instance
caption = moondream_caption("image.tif", length="short")
```

---

## DINOv3 - Feature Extraction & Similarity

**Purpose**: Compute patch-level similarity analysis on geospatial imagery using DINOv3 embeddings

### Initialization

```python
from geoai.dinov3 import DINOv3GeoProcessor

processor = DINOv3GeoProcessor(
    model_name="dinov3_vitl16",  # Model variant
    weights_path=None,            # Optional custom weights
    device=None                   # Auto-detect cuda/cpu
)
```

**Available Models:**
- `dinov3_vitl16` (large)
- `dinov3_vitb16` (base)
- `dinov3_vits16` (small)

### 1. Compute Similarity

Find similar regions in an image based on a query point.

```python
result = processor.compute_similarity(
    source="path/to/image.tif",
    query_coords=(x, y)  # Pixel coordinates
)

# Output dictionary
{
    "similarity_map": np.array(...),  # HxW similarity values
    "query_coords": (x, y),
    "patch_grid": (h_patches, w_patches),
    "metadata": {...}
}
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/dinov3/similarity
Request: { "image": <file>, "query_x": 100, "query_y": 200 }
Response: {
    "similarity_map": [...],
    "query_coords": [100, 200],
    "patch_grid": [32, 32]
}
```

### 2. Extract Features

Get patch-level feature embeddings.

```python
# Load and preprocess
data, metadata = processor.load_image("image.tif")
image = processor.preprocess_image_for_dinov3(data)

# Extract features
features, h_patches, w_patches = processor.extract_features(image)

# features shape: [num_patches, feature_dim]
# Typically: [h_patches * w_patches, 1024] for vitl16
```

### 3. Compute Patch Similarity

Calculate similarity between specific patches.

```python
# Get similarity for a specific patch
similarities = processor.compute_patch_similarity(
    features=features,
    patch_x=10,
    patch_y=15
)

# Returns: similarity array for all patches
```

### 4. Visualization

```python
from geoai.dinov3 import visualize_similarity_results

results = visualize_similarity_results(
    input_image="image.tif",
    query_coords=(x, y),
    overlay=True,        # Overlay similarity on original image
    colormap='jet',      # Color map for visualization
    alpha=0.5            # Transparency for overlay
)
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/dinov3/visualize
Request: { "image": <file>, "query_x": 100, "query_y": 200, "overlay": true }
Response: { "visualization": <base64_image> }
```

### Use Cases for FastAPI

- **Similarity Search**: Find similar land cover patterns
- **Change Detection**: Compare regions across time
- **Feature Extraction**: Extract embeddings for ML pipelines
- **Interactive Exploration**: Click to find similar areas

---

## SamGeo3 - Segmentation Anything Model

**Purpose**: Segment geospatial imagery using text, point, or box prompts

### Initialization

```python
from samgeo import SamGeo3

sam = SamGeo3(
    backend='meta',                      # or 'transformers'
    model_id='facebook/sam3',
    device='cuda',                       # or 'cpu'
    enable_inst_interactivity=False,    # True for point/box prompts
    confidence_threshold=0.5,
    resolution=1008
)
```

**Backend Options:**
- `meta`: Facebook's official implementation
- `transformers`: Hugging Face implementation

### 1. Text-based Segmentation

Segment objects using natural language descriptions.

```python
# Set image first
sam.set_image("path/to/image.tif")

# Generate masks
masks = sam.generate_masks(
    prompt="water",
    min_mask_region_area=100,  # Filter small objects
    max_mask_region_area=10000 # Filter large objects
)

# Save results
sam.save_masks(
    output="water_mask.tif",
    dtype='uint8'
)
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/sam/segment/text
Request: {
    "image": <file>,
    "prompt": "water",
    "min_area": 100,
    "max_area": 10000
}
Response: { "mask": <base64_raster>, "count": 5 }
```

### 2. Point-based Segmentation

Segment using point prompts (interactive).

```python
sam = SamGeo3(enable_inst_interactivity=True)
sam.set_image("image.tif")

# Single foreground point
sam.generate_masks_by_points(
    point_coords=[[520, 375]],
    point_labels=[1]  # 1=foreground, 0=background
)

# Multiple points (refine selection)
sam.generate_masks_by_points(
    point_coords=[[520, 375], [600, 400], [450, 350]],
    point_labels=[1, 1, 0]  # 2 foreground, 1 background
)

# Low-level API for more control
masks = sam.predict_inst(
    point_coords=[[520, 375]],
    point_labels=[1],
    multimask_output=True  # Generate multiple mask options
)
```

**Point Label Meanings:**
- `1` = Foreground (include this region)
- `0` = Background (exclude this region)

**FastAPI Endpoint Mapping:**
```
POST /api/v1/sam/segment/points
Request: {
    "image": <file>,
    "points": [[520, 375], [600, 400]],
    "labels": [1, 0]  # foreground, background
}
Response: { "masks": [...], "scores": [...] }
```

### 3. Box-based Segmentation

Segment using bounding box prompts.

```python
sam = SamGeo3(enable_inst_interactivity=True)
sam.set_image("image.tif")

# Single box: [xmin, ymin, xmax, ymax]
sam.generate_masks_by_boxes(
    boxes=[[100, 100, 500, 500]]
)

# Multiple boxes
sam.generate_masks_by_boxes(
    boxes=[
        [100, 100, 500, 500],
        [600, 600, 900, 900]
    ]
)

# From GeoJSON/Shapefile (with CRS)
sam.generate_masks_by_boxes(
    boxes="boxes.geojson",
    box_crs="EPSG:4326"
)

# Save results
sam.save_masks(
    output="segmented.tif",
    dtype='uint8'
)
```

**FastAPI Endpoint Mapping:**
```
POST /api/v1/sam/segment/boxes
Request: {
    "image": <file>,
    "boxes": [[100, 100, 500, 500], ...],
    "crs": "EPSG:4326"  # optional
}
Response: { "masks": [...], "count": 2 }
```

### 4. Tiled Segmentation (Large Images)

Process large images using sliding window approach.

```python
sam = SamGeo3()

sam.generate_masks_tiled(
    source="large_image.tif",
    prompt="buildings",
    tile_size=1024,        # Size of processing tiles
    overlap=128,           # Overlap between tiles
    min_size=100,          # Min object size (pixels)
    max_size=50000,        # Max object size (pixels)
    output="buildings_mask.tif"
)

# Convert raster to vector
sam.raster_to_vector(
    raster="buildings_mask.tif",
    output="buildings.gpkg",
    simplify_tolerance=0.5  # Simplify geometry
)
```

**Tiling Parameters:**
- `tile_size`: Larger = faster but needs more memory
- `overlap`: Reduces edge artifacts (typical: 10-20% of tile_size)
- Adjust based on GPU memory and image size

**FastAPI Endpoint Mapping:**
```
POST /api/v1/sam/segment/tiled
Request: {
    "image": <file>,
    "prompt": "buildings",
    "tile_size": 1024,
    "overlap": 128
}
Response: {
    "mask": <base64_raster>,
    "vector": <geojson>,
    "count": 150
}
```

### 5. Visualization

```python
# Visualize masks
sam.show_anns(
    cmap='viridis',
    alpha=0.5
)

# Show point prompts
sam.show_points(
    point_coords=[[520, 375]],
    point_labels=[1]
)

# Show box prompts
sam.show_boxes(
    boxes=[[100, 100, 500, 500]]
)
```

### Output Formats

**Raster:**
- GeoTIFF (preserves georeferencing)
- Data types: uint8, uint16, float32

**Vector:**
- GeoPackage (.gpkg)
- GeoJSON (.geojson)
- Shapefile (.shp)

---

## FastAPI Integration Patterns

### 1. Model Loading Strategy

**Singleton Pattern** - Load models once, reuse across requests:

```python
from functools import lru_cache

class ModelManager:
    _moondream = None
    _dinov3 = None
    _sam = None

    @classmethod
    def get_moondream(cls):
        if cls._moondream is None:
            cls._moondream = MoondreamGeo(device="cuda")
        return cls._moondream

    @classmethod
    def get_dinov3(cls):
        if cls._dinov3 is None:
            cls._dinov3 = DINOv3GeoProcessor()
        return cls._dinov3

    @classmethod
    def get_sam(cls):
        if cls._sam is None:
            cls._sam = SamGeo3(device="cuda")
        return cls._sam

# In FastAPI endpoint
@app.post("/api/v1/moondream/caption")
async def caption_image(file: UploadFile):
    model = ModelManager.get_moondream()
    # ... process
```

### 2. File Upload Handling

```python
from fastapi import UploadFile, File
import tempfile
from pathlib import Path

@app.post("/api/v1/sam/segment/text")
async def segment_text(
    file: UploadFile = File(...),
    prompt: str = "water"
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process
        sam = ModelManager.get_sam()
        sam.set_image(tmp_path)
        masks = sam.generate_masks(prompt=prompt)

        # Return results
        return {"status": "success", "masks": masks}

    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
```

### 3. Async Processing for Heavy Operations

```python
from fastapi import BackgroundTasks
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

def process_sam_tiled(image_path: str, prompt: str):
    """Heavy processing in thread pool"""
    sam = ModelManager.get_sam()
    sam.generate_masks_tiled(
        source=image_path,
        prompt=prompt,
        output="result.tif"
    )
    return "result.tif"

@app.post("/api/v1/sam/segment/tiled")
async def segment_tiled(
    file: UploadFile,
    prompt: str,
    background_tasks: BackgroundTasks
):
    # Save file
    tmp_path = await save_upload(file)

    # Process in background
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        process_sam_tiled,
        tmp_path,
        prompt
    )

    # Schedule cleanup
    background_tasks.add_task(cleanup_temp_files, tmp_path, result)

    return {"status": "processing", "result_path": result}
```

### 4. Response Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class CaptionResponse(BaseModel):
    caption: str
    length: str
    confidence: Optional[float] = None

class DetectionResponse(BaseModel):
    detections: List[dict]
    count: int
    format: str = "geojson"

class SegmentationResponse(BaseModel):
    masks: List[dict]
    count: int
    output_format: str = "geotiff"

class SimilarityResponse(BaseModel):
    similarity_map: List[List[float]]
    query_coords: tuple[int, int]
    patch_grid: tuple[int, int]
```

### 5. Error Handling

```python
from fastapi import HTTPException

@app.post("/api/v1/moondream/query")
async def query_image(file: UploadFile, question: str):
    try:
        model = ModelManager.get_moondream()
        result = model.query(question=question, source=file)
        return result

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise HTTPException(status_code=507, detail="GPU memory exhausted")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
```

---

## Common Patterns

### Coordinate Systems

**Pixel Coordinates** (for regular images):
```python
# Origin: top-left (0, 0)
# X: increases right
# Y: increases down
point = [100, 200]  # x=100, y=200
box = [xmin, ymin, xmax, ymax]
```

**Geographic Coordinates** (for GeoTIFF):
```python
# Automatically handled if input is GeoTIFF
# Output preserves CRS
# Can specify CRS: "EPSG:4326" (WGS84), "EPSG:3857" (Web Mercator)
```

### Memory Management

```python
import gc
import torch

def cleanup_gpu():
    """Free GPU memory after heavy operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use in endpoints
@app.post("/api/v1/sam/segment/tiled")
async def segment_tiled(...):
    try:
        # Heavy processing
        result = process_large_image()
        return result
    finally:
        cleanup_gpu()
```

### Image Preprocessing

```python
from PIL import Image
import numpy as np

def prepare_image(file_path: str, max_size: int = 1024):
    """Load and optionally resize image"""
    img = Image.open(file_path)

    # Resize if too large
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return img

def validate_image(file: UploadFile):
    """Validate uploaded image"""
    allowed_types = ["image/tiff", "image/jpeg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, "Invalid image format")

    # Check file size (e.g., 50MB limit)
    if file.size > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50MB)")
```

### Response Compression

```python
import base64
import gzip
from io import BytesIO

def compress_numpy_array(arr: np.ndarray) -> str:
    """Compress numpy array for API response"""
    buffer = BytesIO()
    np.save(buffer, arr)
    compressed = gzip.compress(buffer.getvalue())
    return base64.b64encode(compressed).decode('utf-8')

def decompress_numpy_array(data: str) -> np.ndarray:
    """Decompress numpy array from API response"""
    compressed = base64.b64decode(data)
    decompressed = gzip.decompress(compressed)
    buffer = BytesIO(decompressed)
    return np.load(buffer)
```

### Health Check Endpoints

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/health/models")
async def models_health():
    """Check if models are loaded"""
    return {
        "moondream": ModelManager._moondream is not None,
        "dinov3": ModelManager._dinov3 is not None,
        "sam": ModelManager._sam is not None,
        "gpu_available": torch.cuda.is_available()
    }
```

---

## Quick Reference Table

| Model | Primary Use | Input | Output | Typical Response Time |
|-------|-------------|-------|--------|----------------------|
| **Moondream** | Vision-Language | Image + Text | Text/GeoJSON | 1-3s |
| **DINOv3** | Similarity | Image + Point | Similarity Map | 0.5-2s |
| **SamGeo** (Text) | Segmentation | Image + Prompt | Raster Mask | 2-5s |
| **SamGeo** (Point) | Interactive Seg | Image + Points | Raster Mask | 1-2s |
| **SamGeo** (Box) | Precise Seg | Image + Boxes | Raster Mask | 1-3s |
| **SamGeo** (Tiled) | Large Images | Large Image | Raster/Vector | 10-60s |

---

## Environment Variables

```bash
# Model cache location
export HF_HOME=/path/to/cache
export TORCH_HOME=/path/to/torch/cache

# GPU settings
export CUDA_VISIBLE_DEVICES=0

# Model configurations
export MOONDREAM_MODEL=vikhyatk/moondream2
export SAM_MODEL=facebook/sam3
export DINOV3_MODEL=dinov3_vitl16
```

---

## Useful Resources

- **GeoAI Documentation**: https://github.com/opengeos/geoai
- **SamGeo Documentation**: https://github.com/opengeos/segment-geospatial
- **Moondream Examples**: https://github.com/opengeos/geoai/blob/main/docs/examples/moondream.ipynb
- **DINOv3 Examples**: https://github.com/opengeos/geoai/blob/main/docs/examples/DINOv3_visualization.ipynb
- **SamGeo Examples**: https://github.com/opengeos/segment-geospatial/tree/main/docs/examples

---

**Last Updated**: 2025-12-21
**Version**: 1.0
