#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

# Import PainterlyScript from paint-img.py (hyphen in filename requires importlib)
spec = importlib.util.spec_from_file_location("paint_img", Path(__file__).parent / "paint-img.py")
paint_img = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paint_img)
PainterlyScript = paint_img.PainterlyScript


def main():
    parser = argparse.ArgumentParser(description="Run painterly rendering on an image")
    parser.add_argument("--input", required=True, help="Source image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--brush", default="assets/brush_01.png", help="Brush texture path")
    parser.add_argument("--brush-size", type=int, default=25, help="Brush size in pixels")
    parser.add_argument("--num-strokes", type=int, default=10000, help="Number of strokes")
    parser.add_argument("--step-size", type=int, default=5, help="Step size between stamps")
    parser.add_argument("--threshold", type=float, default=0.95, help="Coverage threshold (0-1)")
    args = parser.parse_args()

    painter = PainterlyScript(args.input, args.brush, brush_size=args.brush_size)
    painter.paint(num_strokes=args.num_strokes, step_size=args.step_size, threshold=args.threshold)
    painter.save_result(args.output)


if __name__ == "__main__":
    main()
