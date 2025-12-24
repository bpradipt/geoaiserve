# GeoAI REST API

A FastAPI-based REST API for geospatial AI model inference, providing endpoints for image segmentation, vision-language tasks, feature extraction, and more.

## Features

- **Multiple Model Support**: SAM, Moondream, DINOv3, GroundedSAM, Detectron2, TIMM
- **Lazy Model Loading**: Models are loaded on-demand to optimize memory usage
- **Flexible Input**: Support for file uploads, URLs, and base64-encoded images
- **Geospatial Formats**: GeoJSON, GeoTIFF, Shapefile, GeoPackage, KML outputs
- **Async Processing**: Support for background job processing (planned)
- **Interactive Docs**: Auto-generated OpenAPI documentation

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd geoaiserve

# Install dependencies
uv sync

# Run the server
uv run uvicorn geoaiserve.main:app --reload
```

The API will be available at `http://localhost:8000`

## Configuration

Configuration is managed through environment variables or a `.env` file:

```bash
# Application
APP_NAME="GeoAI REST API"
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000

# Models
GEOAI_MODELS=sam,moondream,dinov3
DEVICE=cpu  # or cuda, mps
SAM_MODEL_NAME=facebook/sam-vit-huge
MOONDREAM_MODEL_NAME=vikhyatk/moondream2
DINOV3_MODEL_NAME=facebook/dinov2-base

# Storage
STORAGE_BACKEND=local
STORAGE_PATH=/tmp/geoaiserve
MAX_UPLOAD_SIZE=104857600  # 100 MB

# CORS
ENABLE_CORS=true
CORS_ORIGINS=*

# Security
API_KEY_REQUIRED=false
RATE_LIMIT=100/minute

# Logging
LOG_LEVEL=info
```

## API Endpoints

### Common Endpoints

#### Health Check
```bash
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-12-22T12:00:00",
  "models_loaded": ["sam", "moondream"]
}
```

#### List Available Models
```bash
GET /api/v1/models
```

Response:
```json
{
  "models": [
    {
      "model_id": "sam",
      "model_type": "sam",
      "model_name": "facebook/sam-vit-huge",
      "description": "Segment Anything Model for image segmentation",
      "supported_tasks": ["automatic_mask_generation", "prompt_based_segmentation"],
      "device": "cpu",
      "loaded": false
    }
  ],
  "total": 3
}
```

#### Get Model Info
```bash
GET /api/v1/models/{model_id}/info
```

### Interactive Documentation

Visit these URLs when the server is running:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **OpenAPI Schema**: http://localhost:8000/api/v1/openapi.json

## Project Structure

```
geoaiserve/
├── __init__.py
├── main.py                 # FastAPI app entry point
├── config.py               # Configuration management
├── models/                 # Model management
│   ├── __init__.py
│   ├── base.py            # Base model interface
│   └── registry.py        # Model registry with lazy loading
├── routers/               # API routers
│   ├── __init__.py
│   └── common.py          # Health & models endpoints
├── schemas/               # Pydantic models
│   ├── __init__.py
│   ├── common.py          # Common schemas
│   ├── requests.py        # Request schemas
│   └── responses.py       # Response schemas
├── services/              # Business logic
│   ├── __init__.py
│   └── file_handler.py    # File upload/download
├── middleware/            # Custom middleware
├── utils/                 # Utility functions
└── feature_store/         # Feature storage (Zarr)
```

## Development

### Running Tests

```bash
# Install test dependencies
uv add --dev pytest pytest-asyncio httpx

# Run tests
uv run pytest
```

### Code Style

This project follows:
- PEP 8 conventions
- Type hints for all functions
- Async/await patterns for I/O operations
- Pydantic for data validation

### Running in Development Mode

```bash
uv run uvicorn geoaiserve.main:app --reload --log-level debug
```

## Implementation Status

### ✅ Phase 1: Core Infrastructure (Completed)

- [x] FastAPI application setup
- [x] Configuration management with Pydantic Settings
- [x] Model registry with lazy loading
- [x] Request/response schemas
- [x] Health check endpoint
- [x] Models listing endpoint
- [x] File upload/download service
- [x] Error handling and logging

### ✅ Phase 2: Model Integration (Completed)

- [x] SAM endpoints (generate, predict, batch)
- [x] Moondream endpoints (caption, query, detect, point)
- [x] DINOv3 endpoints (features, similarity, batch-similarity)
- [x] Model service implementations
- [x] Full request/response schemas
- [x] Mock model fallbacks for testing

### 📋 Phase 3: Advanced Features (Planned)

- [ ] Async job queue (Celery/RQ)
- [ ] Redis caching
- [ ] GroundedSAM endpoints
- [ ] Detectron2 endpoints
- [ ] TIMM endpoints

### 📋 Phase 4: Production Readiness (Planned)

- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Metrics & monitoring
- [ ] Comprehensive tests
- [ ] Docker deployment
- [ ] CI/CD pipeline

## Architecture

The API follows a unified gateway pattern with:

- **Router Layer**: Organized endpoints per model type
- **Service Layer**: Business logic and model management
- **Model Registry**: Centralized model loading and caching
- **File Handler**: Input/output file management

Models are loaded lazily on first request and cached for subsequent requests.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
