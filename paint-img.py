import cv2
import numpy as np
import random
from skimage.segmentation import slic
from skimage.color import rgb2lab

class PainterlyScript:
    def __init__(self, source_path, brush_path, brush_size=20):
        self.img = cv2.imread(source_path)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.brush_orig = cv2.imread(brush_path, cv2.IMREAD_UNCHANGED)
        self.brush_size = brush_size
        
        # Canvas and Tracking
        self.canvas = np.ones_like(self.img) * 255
        self.coverage_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        
        # Calculate Gradients for direction
        gx = cv2.Sobel(self.img_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.img_gray, cv2.CV_32F, 0, 1, ksize=3)
        self.magnitude, self.angle = cv2.cartToPolar(gx, gy)

    def get_brush_texture(self, color, angle, scale_mod=1.0):
        """Resizes, rotates, and colors the brush texture."""
        size = int(self.brush_size * scale_mod)
        brush = cv2.resize(self.brush_orig, (size, size))
        
        # Rotate brush to match local image flow
        center = (size // 2, size // 2)
        matrix = cv2.getRotationMatrix2D(center, np.degrees(angle) + 90, 1.0)
        brush = cv2.warpAffine(brush, matrix, (size, size))
        
        # Color the brush (assuming brush is grayscale/alpha)
        # If brush has 4 channels, use alpha; otherwise, treat as mask
        if brush.shape[2] == 4:
            alpha = brush[:, :, 3] / 255.0
            colored_brush = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(3):
                colored_brush[:, :, i] = color[i]
            return colored_brush, alpha
        else:
            return brush, None

    def paint(self, num_strokes=5000, step_size=5, threshold=0.95):
        # 1. Cluster image to find regions
        segments = slic(self.img, n_segments=200, compactness=10, start_label=1)
        
        strokes_count = 0
        while strokes_count < num_strokes:
            # 2. Strategy: Sample a random point that isn't covered yet
            y, x = random.randint(0, self.img.shape[0]-1), random.randint(0, self.img.shape[1]-1)
            
            if self.coverage_mask[y, x] > 200: # Threshold for "already painted"
                continue
            
            # 3. Get local properties
            color = self.img[y, x].tolist()
            angle = self.angle[y, x]
            
            # 4. Draw the stroke
            self.apply_stroke(x, y, color, angle, step_size)
            
            strokes_count += 1
            if np.mean(self.coverage_mask) / 255.0 > threshold:
                break

    def apply_stroke(self, x, y, color, angle, length):
        """Simulates a single brush movement."""
        b_tex, alpha = self.get_brush_texture(color, angle)
        h, w = b_tex.shape[:2]
        
        # Boundary checks
        y1, y2 = max(0, y - h//2), min(self.img.shape[0], y + h//2)
        x1, x2 = max(0, x - w//2), min(self.img.shape[1], x + w//2)
        
        # Crop brush if it goes off canvas
        bh1, bh2 = h//2 - (y - y1), h//2 + (y2 - y)
        bw1, bw2 = w//2 - (x - x1), w//2 + (x2 - x)
        
        # Blend
        if alpha is not None:
            region = self.canvas[y1:y2, x1:x2]
            alpha_crop = alpha[bh1:bh2, bw1:bw2, np.newaxis]
            self.canvas[y1:y2, x1:x2] = (1 - alpha_crop) * region + alpha_crop * b_tex[bh1:bh2, bw1:bw2]
            
            # Update coverage mask
            cv2.circle(self.coverage_mask, (x, y), self.brush_size // 2, 255, -1)

    def save_result(self, output_path):
        cv2.imwrite(output_path, self.canvas)

# Usage
# painter = PainterlyScript('input.jpg', 'brush_texture.png', brush_size=25)
# painter.paint(num_strokes=10000)
# painter.save_result('output.jpg')