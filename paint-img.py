import cv2
import numpy as np
import random
from skimage.segmentation import slic
from skimage.measure import regionprops
from tqdm import tqdm


class PainterlyScript:
    def __init__(self, source_path, brush_path, brush_size=20, bg_color="#000000", bg_threshold=10):
        self.img = cv2.imread(source_path)
        if self.img is None:
            raise ValueError(f"Could not load image: {source_path}")
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.brush_orig = cv2.imread(brush_path, cv2.IMREAD_UNCHANGED)
        if self.brush_orig is None:
            raise ValueError(f"Could not load brush: {brush_path}")
        self.brush_size = brush_size

        # Parse background color (hex to BGR)
        self.bg_color = self._parse_hex_color(bg_color)
        self.bg_threshold = bg_threshold

        # Precompute background mask for efficiency
        self.bg_mask = self._compute_bg_mask()

        # Canvas starts as copy of source (not white!)
        self.canvas = self.img.copy()
        self.coverage_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)

        # Calculate Gradients for stroke direction
        gx = cv2.Sobel(self.img_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.img_gray, cv2.CV_32F, 0, 1, ksize=3)
        self.magnitude, self.angle = cv2.cartToPolar(gx, gy)

        # Default parameters
        self.step_size = 5
        self.color_threshold = 50
        self.max_stroke_length = 20

        # Jitter settings
        self.jitter_size = 0.2
        self.jitter_angle = 15
        self.jitter_opacity = 0.3

    def _parse_hex_color(self, hex_color):
        """Parse hex color string to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.float32)  # BGR for OpenCV

    def _compute_bg_mask(self):
        """Precompute mask of background pixels."""
        diff = np.linalg.norm(self.img.astype(np.float32) - self.bg_color, axis=2)
        return (diff < self.bg_threshold).astype(np.uint8) * 255

    def is_background(self, x, y):
        """Check if pixel is background."""
        return self.bg_mask[y, x] > 0

    def get_brush_texture(self, color, angle, scale_mod=1.0):
        """Resizes, rotates, and colors the brush texture with jitter."""
        # Apply size jitter
        jitter = 1.0 + random.uniform(-self.jitter_size, self.jitter_size)
        size = max(4, int(self.brush_size * scale_mod * jitter))
        brush = cv2.resize(self.brush_orig, (size, size))

        # Apply angle jitter
        angle_jitter = random.uniform(-self.jitter_angle, self.jitter_angle)
        final_angle = np.degrees(angle) + 90 + angle_jitter

        # Rotate brush to match local image flow
        center = (size // 2, size // 2)
        matrix = cv2.getRotationMatrix2D(center, final_angle, 1.0)
        brush = cv2.warpAffine(brush, matrix, (size, size))

        # Color the brush
        if len(brush.shape) == 3 and brush.shape[2] == 4:
            alpha = brush[:, :, 3] / 255.0
            # Apply opacity jitter
            opacity_jitter = 1.0 - random.uniform(0, self.jitter_opacity)
            alpha = alpha * opacity_jitter

            colored_brush = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(3):
                colored_brush[:, :, i] = color[i]
            return colored_brush, alpha
        else:
            return brush, None

    def apply_stamp(self, x, y, color, angle, scale_mod=1.0):
        """Apply a single brush stamp at the given position."""
        b_tex, alpha = self.get_brush_texture(color, angle, scale_mod)
        if alpha is None:
            return

        h, w = b_tex.shape[:2]

        # Boundary checks
        y1, y2 = max(0, y - h // 2), min(self.img.shape[0], y + h // 2)
        x1, x2 = max(0, x - w // 2), min(self.img.shape[1], x + w // 2)

        if y2 <= y1 or x2 <= x1:
            return

        # Crop brush if it goes off canvas
        bh1, bh2 = h // 2 - (y - y1), h // 2 + (y2 - y)
        bw1, bw2 = w // 2 - (x - x1), w // 2 + (x2 - x)

        # Ensure valid crop dimensions
        if bh2 <= bh1 or bw2 <= bw1:
            return

        # Blend
        region = self.canvas[y1:y2, x1:x2].astype(np.float32)
        alpha_crop = alpha[bh1:bh2, bw1:bw2, np.newaxis]
        brush_crop = b_tex[bh1:bh2, bw1:bw2].astype(np.float32)

        self.canvas[y1:y2, x1:x2] = (
            (1 - alpha_crop) * region + alpha_crop * brush_crop
        ).astype(np.uint8)

        # Update coverage mask
        cv2.circle(self.coverage_mask, (x, y), int(self.brush_size * scale_mod) // 2, 255, -1)

    def flow_stroke(self, start_x, start_y, scale_mod=1.0):
        """Execute a flowing stroke along the gradient field."""
        x, y = start_x, start_y

        # Skip if starting on background
        if self.is_background(x, y):
            return

        start_color = self.img[y, x].tolist()

        # Apply first stamp
        angle = self.angle[y, x]
        self.apply_stamp(x, y, start_color, angle, scale_mod)

        for step in range(self.max_stroke_length):
            # Get perpendicular to gradient (flow along edges)
            angle = self.angle[y, x] + np.pi / 2

            # Step in flow direction
            dx = int(self.step_size * np.cos(angle))
            dy = int(self.step_size * np.sin(angle))
            new_x, new_y = x + dx, y + dy

            # Bounds check
            if not (0 <= new_x < self.img.shape[1] and 0 <= new_y < self.img.shape[0]):
                break

            # Stop if entering background
            if self.is_background(new_x, new_y):
                break

            # Color difference check - lift brush if color changes too much
            current_color = self.img[new_y, new_x]
            color_diff = np.linalg.norm(
                np.array(current_color, dtype=np.float32) - np.array(start_color, dtype=np.float32)
            )
            if color_diff > self.color_threshold:
                break

            # Apply brush stamp at this position
            self.apply_stamp(new_x, new_y, start_color, angle, scale_mod)
            x, y = new_x, new_y

    def find_next_point_random(self):
        """Find a random uncovered non-background point."""
        for _ in range(1000):
            y = random.randint(0, self.img.shape[0] - 1)
            x = random.randint(0, self.img.shape[1] - 1)
            if self.coverage_mask[y, x] < 200 and not self.is_background(x, y):
                return x, y
        # Fallback: just return random non-background point
        for _ in range(100):
            y = random.randint(0, self.img.shape[0] - 1)
            x = random.randint(0, self.img.shape[1] - 1)
            if not self.is_background(x, y):
                return x, y
        return random.randint(0, self.img.shape[1] - 1), random.randint(0, self.img.shape[0] - 1)

    def find_next_point_distance(self):
        """Find the point furthest from any painted area using distance transform."""
        inverted = 255 - self.coverage_mask
        # Mask out background pixels
        inverted = np.where(self.bg_mask > 0, 0, inverted)

        if np.max(inverted) == 0:
            return self.find_next_point_random()

        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        return int(x), int(y)

    def compute_error_map(self):
        """Compute per-pixel error between canvas and source."""
        diff = cv2.absdiff(self.canvas, self.img)
        return cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    def find_next_point_error(self):
        """Find the point with highest error (biggest difference from source)."""
        error_map = self.compute_error_map()
        # Mask out already well-covered areas slightly
        error_map = error_map.astype(np.float32) * (1 - self.coverage_mask.astype(np.float32) / 255 * 0.5)
        # Mask out background pixels
        error_map = np.where(self.bg_mask > 0, 0, error_map)
        y, x = np.unravel_index(np.argmax(error_map), error_map.shape)
        return int(x), int(y)

    def paint_flow(self, num_strokes, scale_mod=1.0, use_distance=True):
        """Particle Flow painting strategy."""
        for i in tqdm(range(num_strokes), desc="  Flow", leave=False):
            if use_distance and i % 10 == 0:
                x, y = self.find_next_point_distance()
            else:
                x, y = self.find_next_point_random()
            self.flow_stroke(x, y, scale_mod)

    def paint_error(self, num_strokes, scale_mod=1.0):
        """Error-driven painting strategy - focus on high-difference areas."""
        for _ in tqdm(range(num_strokes), desc="  Error", leave=False):
            x, y = self.find_next_point_error()
            self.flow_stroke(x, y, scale_mod)

    def paint_superpixel(self, num_strokes, scale_mod=1.0):
        """Superpixel seeding strategy - paint from SLIC cluster centroids."""
        # Compute SLIC segments
        n_segments = min(num_strokes, 500)
        segments = slic(self.img, n_segments=n_segments, compactness=10, start_label=0)

        # Get region centroids
        props = regionprops(segments + 1)  # +1 because regionprops expects 1-indexed
        centroids = [(int(p.centroid[1]), int(p.centroid[0])) for p in props]

        # Shuffle for more natural look
        random.shuffle(centroids)

        for x, y in tqdm(centroids[:num_strokes], desc="  Superpixel", leave=False):
            if 0 <= x < self.img.shape[1] and 0 <= y < self.img.shape[0]:
                # Skip background centroids
                if not self.is_background(x, y):
                    self.flow_stroke(x, y, scale_mod)

    def paint(self, strategy="flow", num_strokes=5000, passes=3, step_size=5,
              color_threshold=50, max_stroke_length=20,
              jitter_size=0.2, jitter_angle=15, jitter_opacity=0.3):
        """
        Main painting method with multi-pass support.

        Args:
            strategy: 'flow', 'error', or 'superpixel'
            num_strokes: Total strokes (divided among passes)
            passes: Number of refinement passes (1-3)
            step_size: Distance between stamps in a stroke
            color_threshold: Max color diff before lifting brush
            max_stroke_length: Max stamps per stroke
            jitter_*: Randomization parameters
        """
        # Store parameters
        self.step_size = step_size
        self.color_threshold = color_threshold
        self.max_stroke_length = max_stroke_length
        self.jitter_size = jitter_size
        self.jitter_angle = jitter_angle
        self.jitter_opacity = jitter_opacity

        # Multi-pass brush sizes: large -> medium -> small
        pass_scales = {
            1: [1.0],
            2: [2.0, 1.0],
            3: [3.0, 1.5, 1.0],
        }
        scales = pass_scales.get(passes, [1.0])
        strokes_per_pass = num_strokes // len(scales)

        strategy_funcs = {
            "flow": self.paint_flow,
            "error": self.paint_error,
            "superpixel": self.paint_superpixel,
        }
        paint_func = strategy_funcs.get(strategy, self.paint_flow)

        for i, scale in enumerate(scales):
            print(f"Pass {i + 1}/{len(scales)}: brush scale {scale}x, {strokes_per_pass} strokes")
            if strategy == "flow":
                paint_func(strokes_per_pass, scale_mod=scale, use_distance=(i == 0))
            else:
                paint_func(strokes_per_pass, scale_mod=scale)

    def save_result(self, output_path):
        cv2.imwrite(output_path, self.canvas)
        print(f"Saved to {output_path}")


# Usage example
# painter = PainterlyScript('input.jpg', 'brush_texture.png', brush_size=25)
# painter.paint(strategy='flow', num_strokes=10000, passes=3)
# painter.save_result('output.jpg')
