.PHONY: help install dev test test-cov test-watch lint format clean run docs

# Default target
help:
	@echo "GeoAI REST API - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install     Install dependencies"
	@echo "  dev         Start development server with hot reload"
	@echo "  run         Start production server"
	@echo "  test        Run all tests"
	@echo "  test-cov    Run tests with coverage report"
	@echo "  test-watch  Run tests in watch mode (requires pytest-watch)"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with ruff"
	@echo "  clean       Remove build artifacts and cache"
	@echo "  docs        Open API documentation in browser"

# Install dependencies
install:
	uv sync

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
