"""Pytest markers and skip decorators for real model testing."""

from __future__ import annotations

import pytest

# Check if ML dependencies are available
try:
    import torch
    import transformers

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Check if geoai is available
try:
    import geoai

    GEOAI_AVAILABLE = True
except ImportError:
    GEOAI_AVAILABLE = False

# Check if samgeo is available
try:
    import samgeo

    SAMGEO_AVAILABLE = True
except ImportError:
    SAMGEO_AVAILABLE = False

# Skip decorators
skip_without_ml = pytest.mark.skipif(
    not ML_AVAILABLE,
    reason="ML dependencies (torch, transformers) not installed. Install with: uv sync --group ml",
)

skip_without_samgeo = pytest.mark.skipif(
    not SAMGEO_AVAILABLE,
    reason="samgeo not installed. Install with: pip install samgeo",
)

skip_without_torch = pytest.mark.skipif(
    not ML_AVAILABLE,
    reason="PyTorch not installed. Install with: uv sync --group ml",
)

skip_without_geoai = pytest.mark.skipif(
    not GEOAI_AVAILABLE,
    reason="geoai not installed. Install with: uv sync --group ml",
)


def requires_geotiff_data(test_data_dir):
    """Skip test if no GeoTIFF files are available in test data directory."""
    from pathlib import Path

    data_dir = Path(test_data_dir) if isinstance(test_data_dir, str) else test_data_dir
    geotiff_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))
    return pytest.mark.skipif(
        len(geotiff_files) == 0,
        reason=f"No GeoTIFF files found in {data_dir}. Add .tif/.tiff files to run this test.",
    )
