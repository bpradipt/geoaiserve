# Test Data Directory

This directory contains GeoTIFF files for real model testing.

## Requirements

Place your GeoTIFF test files (`.tif` or `.tiff`) in this directory to run real model tests.

### Recommended Test Files

- **Small satellite images** (< 50MB) for quick testing
- **Multi-band GeoTIFF** files (3+ bands) for RGB conversion testing
- **Single-band GeoTIFF** files for grayscale handling

## File Guidelines

- Files should be valid GeoTIFF format with georeferencing metadata
- Recommended size: 512x512 to 2048x2048 pixels for reasonable test times
- Various data types supported: uint8, uint16, int16, float32

## Git Configuration

GeoTIFF files are excluded from version control via `.gitignore`:

```
tests/data/*.tif
tests/data/*.tiff
```

The `.gitkeep` and `README.md` files are tracked to maintain directory structure.

## Running Tests

```bash
# Copy your test files
cp /path/to/your/satellite.tif tests/data/

# Run real model tests
pytest -m real_model -v

# Run GeoTIFF-specific tests
pytest -m geotiff -v
```

## Sample Sources

Free GeoTIFF data sources for testing:

- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- [NASA Earthdata](https://earthdata.nasa.gov/)
- [OpenAerialMap](https://openaerialmap.org/)
