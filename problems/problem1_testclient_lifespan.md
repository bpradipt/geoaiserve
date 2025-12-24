# Problem 1: TestClient Doesn't Trigger Lifespan Events

## Issue

When running pytest tests, all model endpoints were returning 500 errors:
```
Model type ModelType.SAM not registered. Available types: []
```

Models were not being registered even though the registration code was in the `lifespan` function.

## Root Cause

FastAPI's `TestClient` does not automatically trigger lifespan events (startup/shutdown) unless used as a context manager.

## Code Before Fix

```python
@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(app)
```

## Solution

Use `TestClient` as a context manager to ensure lifespan events are triggered:

```python
@pytest.fixture
def client():
    """Create FastAPI test client with lifespan events."""
    with TestClient(app) as client:
        yield client
```

## Verification

After the fix:
- Models are registered during test setup
- Lifespan startup events execute
- All 7 common endpoint tests pass ✓

## Files Changed

- `tests/conftest.py` - Updated client fixture

## Lesson Learned

Always use `with TestClient(app) as client:` when testing FastAPI apps that rely on lifespan events for initialization (model loading, database connections, etc.).

## Test Results

Before: 7/36 passed (only common endpoints worked)
After: 19/36 passed (model endpoints now work)
