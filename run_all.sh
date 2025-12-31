#!/bin/bash

# Default parameters
BRUSH="assets/brush_01.png"
BRUSH_SIZE=120
NUM_STROKES=10000
STEP_SIZE=5
THRESHOLD=0.95

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --brush) BRUSH="$2"; shift 2 ;;
        --brush-size) BRUSH_SIZE="$2"; shift 2 ;;
        --num-strokes) NUM_STROKES="$2"; shift 2 ;;
        --step-size) STEP_SIZE="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
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
        --num-strokes "$NUM_STROKES" \
        --step-size "$STEP_SIZE" \
        --threshold "$THRESHOLD"

    # Log the run
    echo "[$timestamp] input=$basename output=$(basename "$output") brush=$BRUSH brush_size=$BRUSH_SIZE num_strokes=$NUM_STROKES step_size=$STEP_SIZE threshold=$THRESHOLD" >> "$LOG_FILE"
done

echo "Done. See $LOG_FILE for run details."
