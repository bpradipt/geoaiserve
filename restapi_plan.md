# GeoAI REST API Architecture Proposal

## Current State

**Available Models:**
- **SAM** (Segment Anything Model) - Segmentation
- **Moondream** - Vision-language model (VQA, captioning, detection)
- **DINOv3** - Feature extraction and similarity analysis
- **GroundedSAM** - Text-prompted segmentation
- **Detectron2** - Instance segmentation
- **TIMM** - Classification and segmentation models
- **AutoGeoModel** - Universal HuggingFace model interface

## Recommended REST API Architecture

### **Option 1: Unified API Gateway (Recommended)**

A single FastAPI service that exposes all models through organized endpoints:

```
Architecture:
┌─────────────────────────────────────────┐
│  FastAPI Gateway (geoaiserve.main:app) │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Router Layer                   │   │
│  │  /sam/*    /moondream/*  /dino/* │   │
│  │  /grounded-sam/*   /detectron2/* │   │
│  └─────────────────────────────────┘   │
│             ↓                           │
│  ┌─────────────────────────────────┐   │
│  │   Service Layer                  │   │
│  │  (Model Initialization & Cache)  │   │
│  └─────────────────────────────────┘   │
│             ↓                           │
│  ┌─────────────────────────────────┐   │
│  │   geoai Model Classes            │   │
│  │  SamGeo, MoondreamGeo, DINOv3... │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
         ↓
   Redis Cache (optional)
   File Storage (S3/local)
```

**Endpoint Structure:**

```python
# SAM Endpoints
POST /api/v1/sam/generate              # Automatic mask generation
POST /api/v1/sam/predict               # Prompt-based segmentation
POST /api/v1/sam/batch                 # Batch processing

# Moondream Endpoints
POST /api/v1/moondream/caption         # Image captioning
POST /api/v1/moondream/query           # Visual QA
POST /api/v1/moondream/detect          # Object detection
POST /api/v1/moondream/point           # Point detection

# DINOv3 Endpoints
POST /api/v1/dinov3/features           # Feature extraction
POST /api/v1/dinov3/similarity         # Patch similarity
POST /api/v1/dinov3/batch-similarity   # Batch similarity analysis

# GroundedSAM Endpoints
POST /api/v1/grounded-sam/segment      # Text-prompted segmentation

# Detectron2 Endpoints
POST /api/v1/detectron2/segment        # Instance segmentation

# TIMM Endpoints
POST /api/v1/timm/classify             # Classification
POST /api/v1/timm/segment              # Segmentation

# AutoGeoModel (Universal Interface)
POST /api/v1/auto/infer                # Auto-detect task and run

# Utility Endpoints
GET  /api/v1/health                    # Health check
GET  /api/v1/models                    # List available models
GET  /api/v1/models/{model_id}/info    # Model details
POST /api/v1/jobs                      # Async job submission
GET  /api/v1/jobs/{job_id}            # Job status
GET  /api/v1/jobs/{job_id}/result     # Download result
```

**Request/Response Format:**

```python
# Request (multipart/form-data)
{
    "image": <file>,                    # Or URL or base64
    "model_params": {
        "model_name": "facebook/sam-vit-huge",
        "device": "cuda"
    },
    "task_params": {
        # Task-specific parameters
        "points": [[100, 200]],
        "point_labels": [1]
    },
    "output": {
        "format": "geojson|geotiff|shapefile|json",
        "simplify": 0.001,
        "crs": "EPSG:4326"
    }
}

# Response
{
    "status": "success",
    "job_id": "uuid-123",             # For async jobs
    "result": {
        "type": "FeatureCollection",   # For GeoJSON
        "features": [...],
        "metadata": {
            "crs": "EPSG:4326",
            "bounds": [...],
            "processing_time": 2.5
        }
    },
    "download_url": "/downloads/uuid-123.geojson"  # For file downloads
}
```

### **Option 2: Microservices Architecture**

Separate FastAPI services for each model/category:

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ SAM Service  │  │ Moondream    │  │ DINOv3       │
│   :8001      │  │ Service      │  │ Service      │
│              │  │   :8002      │  │   :8003      │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                 ↓                 ↓
┌─────────────────────────────────────────────────┐
│      API Gateway / Load Balancer                │
│         (Nginx / Traefik / Kong)                │
└─────────────────────────────────────────────────┘
```

**Pros:**
- Independent scaling per model
- Isolated failures
- Different resource requirements (GPU/CPU)

**Cons:**
- More complex deployment
- Higher overhead
- Network latency between services

---

## Implementation Recommendations

### **1. Core Service Structure** (geoaiserve/)

```
geoaiserve/
├── __init__.py
├── main.py                 # FastAPI app entry point
├── config.py               # Configuration management
├── models/                 # Model management
│   ├── __init__.py
│   ├── base.py            # Base model interface
│   ├── sam_service.py
│   ├── moondream_service.py
│   ├── dinov3_service.py
│   ├── grounded_sam_service.py
│   └── registry.py        # Model registry/factory
├── routers/               # API routers
│   ├── __init__.py
│   ├── sam.py
│   ├── moondream.py
│   ├── dinov3.py
│   └── common.py          # Health, models list
├── schemas/               # Pydantic models
│   ├── __init__.py
│   ├── requests.py        # Request schemas
│   ├── responses.py       # Response schemas
│   └── common.py
├── services/              # Business logic
│   ├── __init__.py
│   ├── file_handler.py    # File upload/download
│   ├── cache.py           # Redis caching
│   ├── job_queue.py       # Async job processing (Celery/RQ)
│   └── storage.py         # S3/local file storage
├── feature_store/
│   │   ├── __init__.py
│   │   ├── base.py          # AbstractFeatureStore
│   │   ├── zarr_store.py    # ZarrFeatureStore
│   │   ├── index.py         # Spatial + vector index helpers
├── middleware/
│   ├── __init__.py
│   ├── auth.py            # Authentication
│   ├── rate_limit.py      # Rate limiting
│   └── monitoring.py      # Prometheus metrics
└── utils/
    ├── __init__.py
    ├── geospatial.py      # GeoTIFF helpers
    └── validation.py      # Input validation
```

### **2. Key Design Patterns**

**Model Registry Pattern:**
```python
class ModelRegistry:
    """Centralized model management with lazy loading"""

    _models = {}

    @classmethod
    def get_model(cls, model_type: str, config: dict):
        """Get or initialize model with caching"""
        key = f"{model_type}_{hash(frozenset(config.items()))}"
        if key not in cls._models:
            cls._models[key] = cls._initialize_model(model_type, config)
        return cls._models[key]
```

**Async Job Pattern:**
```python
# For long-running inference
@router.post("/sam/generate/async")
async def generate_masks_async(request: SamRequest):
    job_id = await job_queue.enqueue(
        task="sam_generate",
        params=request.dict()
    )
    return {"job_id": job_id, "status": "queued"}

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    status = await job_queue.get_status(job_id)
    return status
```

**Streaming Response Pattern:**
```python
# For large GeoTIFF outputs
@router.post("/sam/generate/download")
async def download_result(request: SamRequest):
    def generate():
        # Process and yield chunks
        for chunk in process_geotiff(request):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="image/tiff",
        headers={"Content-Disposition": "attachment; filename=result.tif"}
    )
```

### **3. Geospatial Data Handling**

**Input Formats:**
- File upload (multipart/form-data)
- URL reference (fetch remote GeoTIFF)
- Base64 encoded data
- Cloud storage path (S3, GCS)

**Output Formats:**
- JSON (pixel coordinates, metadata)
- GeoJSON (georeferenced vectors)
- GeoTIFF (raster outputs)
- Shapefile (zipped)
- GeoPackage
- KML/KMZ

**Metadata Preservation:**
```python
class GeoResponse(BaseModel):
    """Standard geospatial response"""
    result: Union[Dict, List]
    metadata: GeoMetadata
    crs: str
    bounds: List[float]
    transform: Optional[List[float]]

class GeoMetadata(BaseModel):
    processing_time: float
    model_name: str
    input_crs: str
    output_crs: str
    pixel_size: Optional[Tuple[float, float]]
```

### **4. Scalability Considerations**

**Model Loading Strategy:**
```python
# Lazy loading on first request
# Pre-load common models on startup
# LRU cache for model instances
# Separate GPU/CPU model pools
```

**Caching Strategy:**
```python
# Redis for:
# - Job status tracking
# - Request-level caching
# - Temporary inference results
# - Common query results
# - Job status tracking
```

**Queue Strategy:**
```python
# Celery/RQ for:
# - Batch processing
# - Long-running segmentation tasks
# - Large GeoTIFF processing
# - Model fine-tuning jobs (if supported)
```

**Resource Management:**
```python
# GPU queue management
# Concurrent request limits per model
# Memory-based request throttling
# Automatic model offloading when idle
```

### **5. Deployment Configuration**

**Environment Variables:**
```bash
# Model Configuration
GEOAI_MODELS=sam,moondream,dinov3  # Comma-separated
SAM_MODEL_NAME=facebook/sam-vit-huge
MOONDREAM_MODEL_NAME=vikhyatk/moondream2

# Device Management
GEOAI_DEVICE=cuda  # cuda, cpu, mps
GPU_MEMORY_LIMIT=8GB

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4
ENABLE_CORS=true
CORS_ORIGINS=*

# Storage
STORAGE_BACKEND=local  # local, s3, gcs
STORAGE_PATH=/data
S3_BUCKET=geoai-results

# Caching
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# Job Queue
CELERY_BROKER=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Security
API_KEY_REQUIRED=false
API_KEYS=key1,key2,key3
RATE_LIMIT=100/minute

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=info
```

### **6. Docker Deployment**

**Multi-Model Support:**
```yaml
# docker-compose.yml
services:
  geoaiserve-sam:
    build: .
    environment:
      - GEOAI_MODELS=sam
      - GEOAI_DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  geoaiserve-moondream:
    build: .
    environment:
      - GEOAI_MODELS=moondream
      - GEOAI_DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

---

## Specific Model API Recommendations

### **SAM API:**
```python
POST /api/v1/sam/generate
- Automatic mask generation
- Parameters: points_per_side, pred_iou_thresh, stability_score_thresh
- Output: GeoTIFF masks or GeoJSON polygons

POST /api/v1/sam/predict
- Prompt-based segmentation
- Parameters: points, point_labels, boxes
- Output: Single or multiple masks
```

### **Moondream API:**
```python
POST /api/v1/moondream/caption
- Parameters: length (short/normal/long), reasoning
- Output: Text caption

POST /api/v1/moondream/query
- Parameters: question
- Support pre-encoded images for multi-query efficiency
- Output: Text answer

POST /api/v1/moondream/detect
- Parameters: object_type, confidence_threshold
- Output: Bounding boxes as GeoJSON or GeoTIFF
```

### **DINOv3 API:**
```python
POST /api/v1/dinov3/similarity
- Parameters: query_points (coordinates), model_name
- Output: Similarity heatmap as GeoTIFF or JSON

POST /api/v1/dinov3/features
- Extract patch features for custom analysis
- Support tiling for large imagery
- Store features in Zarr for later retrieval
- Output: Feature vectors as JSON/numpy

```

### **GroundedSAM API:**
```python
POST /api/v1/grounded-sam/segment
- Parameters: text_prompt, box_threshold, text_threshold
- Support tiling for large imagery
- Output: Segmentation masks with detections
```

---

## Security & Production Considerations

1. **Authentication**: API keys, JWT tokens, OAuth2
2. **Rate Limiting**: Per-user, per-endpoint limits
3. **Input Validation**: File size limits, format validation, malicious file scanning
4. **Monitoring**: Prometheus metrics, request logging, error tracking
5. **Resource Limits**: Request timeout, memory limits, concurrent request caps
6. **Data Privacy**: Auto-delete uploaded files, encryption at rest

---

## Next Steps to Implementation

1. **Create `geoaiserve/` package** with FastAPI structure
2. **Implement model registry** for centralized model management
3. **Build routers** for each model starting with SAM/Moondream
4. **Add async job queue** for long-running tasks
5. **Implement file storage** backend (local/S3)
6. **Add caching layer** with Redis
7. **Create comprehensive tests** for each endpoint
8. **Documentation** with OpenAPI/Swagger
9. **Deployment scripts** and Docker improvements
10. **Monitoring and observability** setup

---

## Implementation Priority

### Phase 1: Core Infrastructure
- FastAPI application setup
- Model registry implementation
- Basic router structure for health/models endpoints
- File upload/download handling

### Phase 2: Model Integration
- SAM endpoints (most commonly used)
- Moondream endpoints (VQA, captioning)
- Basic error handling and validation

### Phase 3: Advanced Features
- Async job queue for long-running tasks
- Redis caching for performance
- DINOv3 and GroundedSAM endpoints

### Phase 4: Production Readiness
- Authentication and authorization
- Rate limiting
- Monitoring and metrics
- Comprehensive testing
- Documentation

### Phase 5: Optimization
- Model caching strategies
- Request batching
- GPU resource management
- Performance tuning
