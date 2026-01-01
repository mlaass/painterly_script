#!/bin/bash

# Default parameters
BRUSH="assets/brush_01.png"
BRUSH_SIZE=25
STRATEGY="flow"
PASSES=3
NUM_STROKES=5000
STEP_SIZE=5
MAX_STROKE_LENGTH=20
COLOR_THRESHOLD=50
JITTER_SIZE=0.2
JITTER_ANGLE=15
JITTER_OPACITY=0.3
BG_COLOR="#000000"
BG_THRESHOLD=10

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --brush) BRUSH="$2"; shift 2 ;;
        --brush-size) BRUSH_SIZE="$2"; shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        --passes) PASSES="$2"; shift 2 ;;
        --num-strokes) NUM_STROKES="$2"; shift 2 ;;
        --step-size) STEP_SIZE="$2"; shift 2 ;;
        --max-stroke-length) MAX_STROKE_LENGTH="$2"; shift 2 ;;
        --color-threshold) COLOR_THRESHOLD="$2"; shift 2 ;;
        --jitter-size) JITTER_SIZE="$2"; shift 2 ;;
        --jitter-angle) JITTER_ANGLE="$2"; shift 2 ;;
        --jitter-opacity) JITTER_OPACITY="$2"; shift 2 ;;
        --bg-color) BG_COLOR="$2"; shift 2 ;;
        --bg-threshold) BG_THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create output directories
mkdir -p "$SCRIPT_DIR/output"
mkdir -p "$SCRIPT_DIR/logs"

LOG_FILE="$SCRIPT_DIR/logs/run_log.txt"

# Process all images
for img in "$SCRIPT_DIR/images"/*.{png,jpg,jpeg}; do
    [ -e "$img" ] || continue

    basename=$(basename "$img")
    name="${basename%.*}"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output="$SCRIPT_DIR/output/${name}_${timestamp}.png"

    echo "Processing: $basename"

    uv run python "$SCRIPT_DIR/run_painter.py" \
        --input "$img" \
        --output "$output" \
        --brush "$SCRIPT_DIR/$BRUSH" \
        --brush-size "$BRUSH_SIZE" \
        --strategy "$STRATEGY" \
        --passes "$PASSES" \
        --num-strokes "$NUM_STROKES" \
        --step-size "$STEP_SIZE" \
        --max-stroke-length "$MAX_STROKE_LENGTH" \
        --color-threshold "$COLOR_THRESHOLD" \
        --jitter-size "$JITTER_SIZE" \
        --jitter-angle "$JITTER_ANGLE" \
        --jitter-opacity "$JITTER_OPACITY" \
        --bg-color "$BG_COLOR" \
        --bg-threshold "$BG_THRESHOLD"

    # Log the run
    echo "[$timestamp] input=$basename output=$(basename "$output") strategy=$STRATEGY passes=$PASSES brush=$BRUSH brush_size=$BRUSH_SIZE num_strokes=$NUM_STROKES bg_color=$BG_COLOR bg_threshold=$BG_THRESHOLD" >> "$LOG_FILE"
done

echo "Done. See $LOG_FILE for run details."
