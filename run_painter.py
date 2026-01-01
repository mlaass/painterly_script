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

    # Required arguments
    parser.add_argument("--input", required=True, help="Source image path")
    parser.add_argument("--output", required=True, help="Output image path")

    # Brush settings
    parser.add_argument("--brush", default="assets/brush_01.png", help="Brush texture path")
    parser.add_argument("--brush-size", type=int, default=25, help="Base brush size in pixels")

    # Strategy and passes
    parser.add_argument("--strategy", choices=["flow", "error", "superpixel"], default="flow",
                        help="Painting strategy (default: flow)")
    parser.add_argument("--passes", type=int, choices=[1, 2, 3], default=3,
                        help="Number of refinement passes (default: 3)")
    parser.add_argument("--num-strokes", type=int, default=5000, help="Total number of strokes")

    # Stroke parameters
    parser.add_argument("--step-size", type=int, default=5, help="Distance between stamps in a stroke")
    parser.add_argument("--max-stroke-length", type=int, default=20, help="Max stamps per stroke")
    parser.add_argument("--color-threshold", type=float, default=50,
                        help="Color difference to lift brush (0-255)")

    # Jitter parameters
    parser.add_argument("--jitter-size", type=float, default=0.2,
                        help="Size jitter amount (0-1, default: 0.2)")
    parser.add_argument("--jitter-angle", type=float, default=15,
                        help="Angle jitter in degrees (default: 15)")
    parser.add_argument("--jitter-opacity", type=float, default=0.3,
                        help="Opacity jitter amount (0-1, default: 0.3)")

    # Background ignore
    parser.add_argument("--bg-color", default="#000000",
                        help="Background color to ignore (hex, default: #000000)")
    parser.add_argument("--bg-threshold", type=float, default=10,
                        help="Color distance threshold for background detection (default: 10)")

    args = parser.parse_args()

    painter = PainterlyScript(
        args.input, args.brush,
        brush_size=args.brush_size,
        bg_color=args.bg_color,
        bg_threshold=args.bg_threshold
    )
    painter.paint(
        strategy=args.strategy,
        num_strokes=args.num_strokes,
        passes=args.passes,
        step_size=args.step_size,
        color_threshold=args.color_threshold,
        max_stroke_length=args.max_stroke_length,
        jitter_size=args.jitter_size,
        jitter_angle=args.jitter_angle,
        jitter_opacity=args.jitter_opacity,
    )
    painter.save_result(args.output)


if __name__ == "__main__":
    main()
