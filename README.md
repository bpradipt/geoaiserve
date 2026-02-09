# GeoAI REST API

A FastAPI-based REST API for geospatial AI model inference, providing endpoints for image segmentation, vision-language tasks, feature extraction, and more.

This relies on [GeoAI APIs](https://github.com/opengeos/geoai). Kudos to the developers.

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
API_KEYS=              # comma-separated list of valid keys
RATE_LIMIT=100/minute

# Concurrency
MAX_CONCURRENT_INFERENCE=1  # max simultaneous inference requests

# Logging
LOG_LEVEL=info
```

## Security

API key authentication is **disabled by default**. To enable it, set the following environment variables:

```bash
API_KEY_REQUIRED=true
API_KEYS=key1,key2,key3
```

When enabled, every request must include a valid key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: key1" http://localhost:8000/api/v1/health
```

Requests with a missing or invalid key receive a `401 Unauthorized` response.

**CORS and credentials:** If `CORS_CREDENTIALS=true` (the default) and `CORS_ORIGINS=*`, the server automatically sets `allow_credentials=False` to comply with the CORS specification, which forbids credentialed requests with a wildcard origin. A warning is logged when this auto-correction occurs.

## Concurrency

Inference endpoints (SAM, Moondream, DINOv3) are gated by a semaphore that limits the number of requests processed simultaneously. By default only **one** inference request runs at a time. If a request arrives while the limit is reached, it immediately receives a `503 Service Unavailable` response with the message *"Server is busy processing another request"*.

Adjust the limit with the `MAX_CONCURRENT_INFERENCE` environment variable:

```bash
MAX_CONCURRENT_INFERENCE=2  # allow two concurrent inference requests
```

Non-inference endpoints (health checks, model listing, file uploads/downloads) are **not** affected by this limit and remain available regardless of inference load.

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

The project includes comprehensive API contract tests using pytest.

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_sam.py

# Run specific test
uv run pytest tests/test_sam.py::test_sam_generate_endpoint

# Run tests matching a pattern
uv run pytest -k "validation"

# Run tests with coverage report
uv run pytest --cov=geoaiserve --cov-report=term-missing

# Run tests in parallel (if pytest-xdist installed)
uv run pytest -n auto
```

### Test Markers

Tests are organized with pytest markers for flexible test selection:

| Marker | Description | Command |
|--------|-------------|---------|
| `mock` | Tests using mock models (fast, no ML deps) | `pytest -m mock` |
| `real_model` | Tests requiring real ML models | `pytest -m real_model` |
| `real_sam` | SAM-specific real model tests | `pytest -m real_sam` |
| `real_moondream` | Moondream-specific real tests | `pytest -m real_moondream` |
| `real_dinov3` | DINOv3-specific real tests | `pytest -m real_dinov3` |
| `geotiff` | Tests using GeoTIFF files | `pytest -m geotiff` |
| `slow` | Long-running tests | `pytest -m "not slow"` |

### Real Model Testing

To run tests with real ML models:

```bash
# Install ML dependencies
uv sync --group ml

# Add GeoTIFF test files
cp /path/to/satellite.tif tests/data/

# Run real model tests
make test-real

# Or run specific model tests
make test-real-sam
make test-real-moondream
make test-real-dinov3
```

### Mock vs Real Model Control

The API includes safeguards to prevent silent fallback to mock models in production:

| Scenario | Behavior |
|----------|----------|
| **Production** (default) | Raises `ImportError` if ML dependencies missing |
| **Tests** (`GEOAI_ALLOW_MOCK=1`) | Falls back to mock if dependencies missing |
| **Explicit** (`allow_mock=False`) | Always raises error if dependencies missing |

**Environment Variable:**

```bash
# Enable mock fallback (for CI/testing without ML deps)
export GEOAI_ALLOW_MOCK=1

# Production (default) - fails loudly if deps missing
unset GEOAI_ALLOW_MOCK
```

**Programmatic Control:**

```python
from geoaiserve.models.moondream_service import MoondreamService
from geoaiserve.schemas.common import DeviceType

# Production (default) - fails if deps missing
service = MoondreamService(device=DeviceType.CPU)

# Explicitly allow mock (testing only)
service = MoondreamService(device=DeviceType.CPU, allow_mock=True)

# Explicitly require real model
service = MoondreamService(device=DeviceType.CPU, allow_mock=False)

# Check if mock is being used
if service.is_mock:
    print("WARNING: Using mock model")
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures (client, sample images, file_ids)
├── markers.py               # Skip decorators for ML dependencies
├── data/                    # User-provided GeoTIFF test files
│   ├── .gitkeep
│   └── README.md
├── test_common.py           # Health check, model listing endpoints
├── test_files.py            # File upload/download CRUD operations
├── test_sam.py              # SAM model endpoints (mock)
├── test_moondream.py        # Moondream model endpoints (mock)
├── test_dinov3.py           # DINOv3 model endpoints (mock)
├── test_integration.py      # Cross-model integration tests
├── test_geotiff_loader.py   # GeoTIFF loader utility tests
├── test_real_sam.py         # Real SAM model tests
├── test_real_moondream.py   # Real Moondream model tests
└── test_real_dinov3.py      # Real DINOv3 model tests
```

### Test Categories

- **Happy path tests**: Verify endpoints work with valid inputs
- **Validation tests**: Test schema validation (422 errors for invalid parameters)
- **Error handling tests**: Test 404 for non-existent resources, 400 for missing fields
- **Response structure tests**: Validate response field types and values
- **Integration tests**: Test file lifecycle and cross-model usage

### Using Make Commands

Common development tasks are available via Make:

```bash
make test          # Run all tests
make test-cov      # Run tests with coverage
make lint          # Run linting
make format        # Format code
make dev           # Start development server
make clean         # Clean build artifacts
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

## Architecture

The API follows a unified gateway pattern with:

- **Router Layer**: Organized endpoints per model type
- **Service Layer**: Business logic and model management
- **Model Registry**: Centralized model loading and caching
- **File Handler**: Input/output file management

Models are loaded lazily on first request and cached for subsequent requests.

## Contributing

Contributions are most welcome. Please open a PR
