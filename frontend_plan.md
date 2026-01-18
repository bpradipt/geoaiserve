# GeoAI API Test Frontend Plan

## Overview
Create a simple, single-file HTML/JS frontend for developers to manually test the GeoAI REST APIs. No build step required - just serve the HTML file.

## Technology Choice
**Vanilla HTML/JS** - Single `index.html` file with embedded CSS and JavaScript.
- No dependencies or build step
- Easy to serve alongside the FastAPI backend
- Can be served from FastAPI's static files

## Location
`/home/ubuntu/geoaiserve/frontend/index.html`

## Features

### 1. API Health & Status Panel
- Health check display (GET `/health`)
- List available models (GET `/models`)
- Show loaded models and their status

### 2. File Upload Section
- Drag-and-drop or click to upload GeoTIFF/images
- Display uploaded file info (file_id, filename, size)
- Show uploaded image preview
- List uploaded files with delete option

### 3. Interactive Image Canvas
- Display uploaded image
- Draw tools:
  - **Point mode**: Click to add foreground/background points
  - **Box mode**: Click-drag to draw bounding boxes
- Clear annotations button
- Visual feedback for drawn annotations

### 4. SAM Segmentation Panel
- **Point/Box Segmentation** (POST `/sam/predict`)
  - Use drawn points/boxes from canvas
  - Display resulting masks overlaid on image
  - Show scores for each mask
- **Text-Prompted Masks** (POST `/sam/generate`)
  - Text input for prompt (e.g., "tree", "building")
  - Min/max size filters
  - Display generated masks

### 5. Moondream VLM Panel
- **Caption** (POST `/moondream/caption`)
  - Length selector (short/normal/long)
  - Display generated caption
- **Query** (POST `/moondream/query`)
  - Text input for question
  - Display answer
- **Detect** (POST `/moondream/detect`)
  - Object type input
  - Confidence threshold slider
  - Display detections on canvas
- **Point** (POST `/moondream/point`)
  - Object description input
  - Display detected points on canvas

### 6. DINOv3 Features Panel
- **Feature Extraction** (POST `/dinov3/features`)
  - Display feature dimensions
  - Show CLS token info
- **Similarity** (POST `/dinov3/similarity`)
  - Use drawn points as query points
  - Display similarity heatmap

### 7. Response Viewer
- Raw JSON response display
- Processing time display
- Error message display
- Download links for generated files

## UI Layout

```
+------------------------------------------+
|  GeoAI API Tester          [Health: OK]  |
+------------------------------------------+
| [Upload] | Models: SAM ✓ Moondream ✓     |
+------------------------------------------+
|                    |                      |
|   Image Canvas     |   API Controls       |
|   - Draw points    |   [SAM] [Moondream]  |
|   - Draw boxes     |   [DINOv3]           |
|                    |                      |
|   [Point] [Box]    |   --- SAM ---        |
|   [Clear]          |   [Predict]          |
|                    |   Prompt: [____]     |
|                    |   [Generate Masks]   |
|                    |                      |
+------------------------------------------+
|  Response JSON / Results                  |
+------------------------------------------+
```

## Implementation Steps

### Step 1: Create base HTML structure
- HTML skeleton with sections for each panel
- Basic CSS for layout (flexbox/grid)
- Responsive design for different screen sizes

### Step 2: Implement file upload
- File input with drag-drop support
- POST to `/api/v1/files/upload`
- Store file_id for subsequent API calls
- Image preview using canvas

### Step 3: Implement interactive canvas
- Canvas element for image display
- Mouse event handlers for drawing
- Point mode: click to add points with labels
- Box mode: mousedown/mousemove/mouseup for rectangles
- Store annotations as arrays

### Step 4: Implement SAM API calls
- `predict()`: Send points/boxes to `/sam/predict`
- `generateMasks()`: Send text prompt to `/sam/generate`
- Overlay masks on canvas with transparency

### Step 5: Implement Moondream API calls
- `caption()`: POST to `/moondream/caption`
- `query()`: POST to `/moondream/query`
- `detect()`: POST to `/moondream/detect`, draw boxes
- `point()`: POST to `/moondream/point`, draw points

### Step 6: Implement DINOv3 API calls
- `extractFeatures()`: POST to `/dinov3/features`
- `computeSimilarity()`: POST to `/dinov3/similarity`

### Step 7: Add response display
- JSON pretty-print panel
- Error handling and display
- Download links

### Step 8: Integrate with FastAPI
- Add static file serving to main.py
- Mount frontend directory

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `frontend/index.html` | Create | Main test UI (single file) |
| `geoaiserve/main.py` | Modify | Add static file mounting |

## Verification

1. Start the server: `uvicorn geoaiserve.main:app --reload`
2. Open `http://localhost:8000/` in browser
3. Test each API:
   - Upload a test image
   - Draw points/boxes on canvas
   - Call SAM predict and verify masks display
   - Call Moondream caption and verify response
   - Check error handling with invalid inputs
