.PHONY: help install install-ml dev test test-cov test-watch test-mock test-real test-real-sam test-real-moondream test-real-dinov3 test-geotiff lint format clean run docs

# Default target
help:
	@echo "GeoAI REST API - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install         Install dependencies"
	@echo "  install-ml      Install dependencies with ML models"
	@echo "  dev             Start development server with hot reload"
	@echo "  run             Start production server"
	@echo "  test            Run all tests"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  test-watch      Run tests in watch mode (requires pytest-watch)"
	@echo "  test-mock       Run only mock tests (fast)"
	@echo "  test-real       Run only real model tests"
	@echo "  test-real-sam   Run real SAM model tests"
	@echo "  test-real-moondream  Run real Moondream model tests"
	@echo "  test-real-dinov3     Run real DINOv3 model tests"
	@echo "  test-geotiff    Run GeoTIFF-specific tests"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with ruff"
	@echo "  clean           Remove build artifacts and cache"
	@echo "  docs            Open API documentation in browser"

# Install dependencies
install:
	uv sync

# Install dependencies with ML models
install-ml:
	uv sync --group ml

# Start development server
dev:
	uv run uvicorn geoaiserve.main:app --reload --log-level debug

# Start production server
run:
	uv run uvicorn geoaiserve.main:app --host 0.0.0.0 --port 8000

# Run all tests
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=geoaiserve --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

# Run tests in watch mode
test-watch:
	uv run pytest-watch

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=tests/test_sam.py"
	@test -n "$(FILE)" && uv run pytest $(FILE) -v || echo "Please specify FILE=<path>"

# Run only mock tests (fast, no ML dependencies required)
test-mock:
	uv run pytest -m mock -v

# Run only real model tests (requires ML dependencies)
test-real:
	uv run pytest -m real_model -v

# Run real SAM model tests
test-real-sam:
	uv run pytest -m real_sam -v

# Run real Moondream model tests
test-real-moondream:
	uv run pytest -m real_moondream -v

# Run real DINOv3 model tests
test-real-dinov3:
	uv run pytest -m real_dinov3 -v

# Run GeoTIFF-specific tests
test-geotiff:
	uv run pytest -m geotiff -v

# Run linting
lint:
	uv run ruff check geoaiserve tests

# Format code
format:
	uv run ruff format geoaiserve tests
	uv run ruff check --fix geoaiserve tests

# Type checking
typecheck:
	uv run mypy geoaiserve

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Open API docs in browser
docs:
	@echo "Starting server and opening docs..."
	@echo "API Docs: http://localhost:8000/api/v1/docs"
	@open http://localhost:8000/api/v1/docs 2>/dev/null || xdg-open http://localhost:8000/api/v1/docs 2>/dev/null || echo "Open http://localhost:8000/api/v1/docs in your browser"

# Quick check before commit
check: lint test
	@echo "All checks passed!"
