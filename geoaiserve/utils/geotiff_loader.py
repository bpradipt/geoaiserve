"""GeoTIFF loading utility with RGB conversion for model inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine


@dataclass
class GeoTIFFData:
    """Container for GeoTIFF data and metadata."""

    image: Image.Image
    metadata: dict[str, Any]
    original_shape: tuple[int, ...]
    dtype: str


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize array values to uint8 range (0-255).

    Args:
        array: Input numpy array of any dtype

    Returns:
        Normalized uint8 numpy array
    """
    if array.dtype == np.uint8:
        return array

    # Handle different dtypes
    if np.issubdtype(array.dtype, np.floating):
        # For floats, assume 0-1 range or normalize to it
        arr_min, arr_max = array.min(), array.max()
        if arr_max > 1.0:
            # Normalize to 0-1 first
            if arr_max != arr_min:
                array = (array - arr_min) / (arr_max - arr_min)
            else:
                array = np.zeros_like(array)
        # Scale to 0-255
        return (array * 255).astype(np.uint8)
    else:
        # For integers, scale based on actual range
        arr_min, arr_max = array.min(), array.max()
        if arr_max == arr_min:
            return np.zeros(array.shape, dtype=np.uint8)
        normalized = (array.astype(np.float64) - arr_min) / (arr_max - arr_min)
        return (normalized * 255).astype(np.uint8)


def load_geotiff_as_rgb(
    file_path: str | Path,
    bands: tuple[int, int, int] | None = None,
    normalize: bool = True,
) -> GeoTIFFData:
    """Load GeoTIFF file and convert to RGB PIL Image.

    Args:
        file_path: Path to GeoTIFF file
        bands: Tuple of 1-indexed band numbers to use as (R, G, B).
               Defaults to (1, 2, 3) for multi-band or (1, 1, 1) for single-band.
        normalize: Whether to normalize values to 0-255 range

    Returns:
        GeoTIFFData containing the RGB image and metadata

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If specified bands are out of range
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")

    with rasterio.open(file_path) as src:
        # Store metadata
        metadata = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs) if src.crs else None,
            "bounds": list(src.bounds) if src.bounds else None,
            "transform": list(src.transform)[:6] if src.transform else None,
            "nodata": src.nodata,
        }
        original_shape = (src.count, src.height, src.width)
        dtype = str(src.dtypes[0])

        # Determine which bands to use
        if bands is None:
            if src.count >= 3:
                bands = (1, 2, 3)
            else:
                # Single band - replicate to RGB
                bands = (1, 1, 1)

        # Validate bands
        for band in bands:
            if band < 1 or band > src.count:
                raise ValueError(
                    f"Band {band} out of range. File has {src.count} band(s)."
                )

        # Read bands
        r = src.read(bands[0])
        g = src.read(bands[1])
        b = src.read(bands[2])

        # Stack into RGB array
        if normalize:
            r = _normalize_to_uint8(r)
            g = _normalize_to_uint8(g)
            b = _normalize_to_uint8(b)

        rgb_array = np.stack([r, g, b], axis=-1)

        # Convert to PIL Image
        image = Image.fromarray(rgb_array, mode="RGB")

        return GeoTIFFData(
            image=image,
            metadata=metadata,
            original_shape=original_shape,
            dtype=dtype,
        )


def save_rgb_as_geotiff(
    image: Image.Image,
    output_path: str | Path,
    reference_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save RGB PIL Image as GeoTIFF with optional georeferencing.

    Args:
        image: PIL Image to save (RGB mode)
        output_path: Path for output GeoTIFF file
        reference_metadata: Optional metadata dict from load_geotiff_as_rgb
                           to preserve georeferencing

    Returns:
        Path to saved GeoTIFF file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert image to numpy array
    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.array(image)

    # Prepare rasterio parameters
    height, width = array.shape[:2]
    count = 3

    # Build profile
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": count,
    }

    # Add georeferencing if available
    if reference_metadata:
        if reference_metadata.get("crs"):
            profile["crs"] = reference_metadata["crs"]
        if reference_metadata.get("transform"):
            profile["transform"] = Affine(*reference_metadata["transform"])

    # Write file
    with rasterio.open(output_path, "w", **profile) as dst:
        # Write each band (rasterio expects bands-first format)
        for i in range(3):
            dst.write(array[:, :, i], i + 1)

    return output_path


def load_geotiff_metadata(file_path: str | Path) -> dict[str, Any]:
    """Load only metadata from GeoTIFF without reading pixel data.

    Args:
        file_path: Path to GeoTIFF file

    Returns:
        Dictionary containing GeoTIFF metadata

    Raises:
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")

    with rasterio.open(file_path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs) if src.crs else None,
            "bounds": list(src.bounds) if src.bounds else None,
            "transform": list(src.transform)[:6] if src.transform else None,
            "nodata": src.nodata,
        }
