# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Painterly-script is a Python tool that procedurally "paints" images using brush textures. It simulates artistic techniques by intelligently determining stroke direction, length, and placement based on image gradients, creating painterly effects from source images.

## Commands

```bash
# Run batch processing on all images in images/
./run_all.sh

# Run with custom parameters
./run_all.sh --brush-size 30 --num-strokes 5000

# Run on a single image
uv run python run_painter.py --input images/example.png --output output/result.png

# All run_painter.py options
uv run python run_painter.py --input FILE --output FILE \
    --brush assets/brush_01.png \
    --brush-size 25 \
    --num-strokes 10000 \
    --step-size 5 \
    --threshold 0.95
```

## Architecture

### Core Components

**paint-img.py** - Main painting engine (`PainterlyScript` class):
- Loads source image and brush texture
- Calculates gradient field using Sobel operators for stroke direction
- Maintains a coverage mask to track painted areas
- Uses SLIC superpixel segmentation for region-aware painting
- Applies strokes with rotation aligned to local image flow

### Key Algorithm Flow

1. **Preprocessing**: Calculate gradient magnitude/angle from grayscale image using Sobel
2. **Painting loop**: Sample random uncovered points, get local color and angle, apply strokes
3. **Stroke application**: Rotate and color brush texture, alpha-blend onto canvas
4. **Termination**: Stop when stroke count reached or coverage threshold exceeded

### Directory Structure

- `assets/` - Brush texture PNGs (with alpha channel)
- `images/` - Source/test images
- `output/` - Generated painted images (created by run_all.sh)
- `logs/` - Run logs with parameters (created by run_all.sh)
- `docs/` - Project documentation and PRDs

## Dependencies

- numpy - Array operations
- opencv-python (cv2) - Image I/O, gradient calculation, transformations
- scikit-image - SLIC superpixel segmentation

## Future Strategies (from PRD)

Three painting strategies planned:
- **Superpixel Seeding**: Sample from SLIC cluster centroids
- **Error-Driven**: Place strokes where canvas-source difference is highest
- **Particle Flow**: Let strokes drift along color contours
