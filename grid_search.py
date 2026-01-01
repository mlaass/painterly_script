#!/usr/bin/env python3
"""
Grid search script to compare different painting strategies and parameters.
Generates a grid of outputs for visual comparison.
"""
import argparse
import itertools
import importlib.util
from pathlib import Path
from datetime import datetime

# Import PainterlyScript
spec = importlib.util.spec_from_file_location("paint_img", Path(__file__).parent / "paint-img.py")
paint_img = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paint_img)
PainterlyScript = paint_img.PainterlyScript


def run_grid_search(input_path, output_dir, brush_path, param_grid, bg_color="#000000", bg_threshold=10):
    """Run painting with all parameter combinations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Running {len(combinations)} combinations...")
    results = []

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))

        # Create descriptive filename
        param_str = "_".join(f"{k[:3]}{v}" for k, v in params.items())
        output_path = output_dir / f"{Path(input_path).stem}_{param_str}.png"

        print(f"\n[{i}/{len(combinations)}] {params}")

        try:
            painter = PainterlyScript(
                input_path,
                brush_path,
                brush_size=params.get("brush_size", 25),
                bg_color=bg_color,
                bg_threshold=bg_threshold
            )
            painter.paint(
                strategy=params.get("strategy", "flow"),
                num_strokes=params.get("num_strokes", 5000),
                passes=params.get("passes", 3),
                step_size=params.get("step_size", 5),
                max_stroke_length=params.get("max_stroke_length", 20),
                color_threshold=params.get("color_threshold", 50),
                jitter_size=params.get("jitter_size", 0.2),
                jitter_angle=params.get("jitter_angle", 15),
                jitter_opacity=params.get("jitter_opacity", 0.3),
            )
            painter.save_result(str(output_path))
            results.append((params, str(output_path), "success"))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((params, None, str(e)))

    # Write summary
    summary_path = output_dir / "grid_search_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Grid Search Results - {datetime.now().isoformat()}\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Brush: {brush_path}\n")
        f.write(f"Background: {bg_color} (threshold: {bg_threshold})\n")
        f.write("=" * 60 + "\n\n")

        for params, output, status in results:
            f.write(f"Parameters: {params}\n")
            f.write(f"Output: {output}\n")
            f.write(f"Status: {status}\n")
            f.write("-" * 40 + "\n")

    print(f"\nDone! {len(results)} images generated in {output_dir}/")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over painting parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick comparison of strategies
  python grid_search.py --input image.png --quick

  # Full grid search
  python grid_search.py --input image.png --strategies flow error superpixel --brush-sizes 20 40 60

  # Custom parameters
  python grid_search.py --input image.png --strategies flow --brush-sizes 30 --passes 1 2 3 --num-strokes 3000 6000
        """
    )

    parser.add_argument("--input", required=True, help="Source image path")
    parser.add_argument("--output-dir", default="output/grid_search", help="Output directory")
    parser.add_argument("--brush", default="assets/brush_01.png", help="Brush texture path")

    # Background ignore
    parser.add_argument("--bg-color", default="#000000",
                        help="Background color to ignore (hex, default: #000000)")
    parser.add_argument("--bg-threshold", type=float, default=10,
                        help="Color distance threshold for background detection (default: 10)")

    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick comparison: 3 strategies x 2 brush sizes")

    # Grid parameters (each can have multiple values)
    parser.add_argument("--strategies", nargs="+", default=["flow"],
                        choices=["flow", "error", "superpixel"])
    parser.add_argument("--brush-sizes", nargs="+", type=int, default=[25])
    parser.add_argument("--passes", nargs="+", type=int, default=[3])
    parser.add_argument("--num-strokes", nargs="+", type=int, default=[5000])
    parser.add_argument("--step-sizes", nargs="+", type=int, default=[5])
    parser.add_argument("--max-stroke-lengths", nargs="+", type=int, default=[20])
    parser.add_argument("--color-thresholds", nargs="+", type=float, default=[50])
    parser.add_argument("--jitter-sizes", nargs="+", type=float, default=[0.2])
    parser.add_argument("--jitter-angles", nargs="+", type=float, default=[15])

    args = parser.parse_args()

    # Build parameter grid
    if args.quick:
        param_grid = {
            "strategy": ["flow", "error", "superpixel"],
            "brush_size": [25, 50],
            "num_strokes": [3000],
            "passes": [3],
        }
    else:
        param_grid = {
            "strategy": args.strategies,
            "brush_size": args.brush_sizes,
            "passes": args.passes,
            "num_strokes": args.num_strokes,
            "step_size": args.step_sizes,
            "max_stroke_length": args.max_stroke_lengths,
            "color_threshold": args.color_thresholds,
            "jitter_size": args.jitter_sizes,
            "jitter_angle": args.jitter_angles,
        }

    # Remove single-value entries to simplify filenames
    param_grid = {k: v for k, v in param_grid.items() if len(v) > 1 or k == "strategy"}

    # Ensure at least strategy is in the grid
    if "strategy" not in param_grid:
        param_grid["strategy"] = args.strategies

    run_grid_search(args.input, args.output_dir, args.brush, param_grid,
                    bg_color=args.bg_color, bg_threshold=args.bg_threshold)


if __name__ == "__main__":
    main()
