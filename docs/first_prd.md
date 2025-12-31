## 1. Project Objective

To create a script that procedurally "paints" an image using a user-supplied brush texture. The system should intelligently determine stroke direction, length, and placement to mimic human artistic techniques, moving away from simple pixel-pushing toward a **vector-field-guided artistic simulation**.

---

## 2. Core Functional Requirements

### 2.1 Brush Engine & Dynamics

* **Texture Support:** Must accept `.png` or `.tif` files with alpha channels for brush transparency.
* **Rotational Alignment:** The brush must rotate dynamically to align with the local image gradient.
* **Dynamics Jitter:** * **Size Jitter:** Random variance in brush scale per stroke.
* **Opacity Jitter:** Variance in paint "thickness" (alpha).
* **Angle Jitter:** Small random offsets from the "perfect" direction to simulate human hand imperfection.



### 2.2 Intelligence & Pathfinding

* **Gradient Vector Field:** The script must calculate the image gradient (using Sobel or Scharr operators). The brush should move **perpendicular** to the gradient (along the edges) to preserve object shapes.
* **Color Sampling:** Sample the source image at the start of a stroke. If the stroke length increases, continue sampling to decide when to "lift the brush" (i.e., when the color difference  exceeds a threshold).
* **Spatial Occupancy (Coverage Mask):** A 1-bit or 8-bit mask tracks where paint has been applied.
* **Proximity Logic:** New strokes should prioritize areas with low coverage (the "white space").



---

## 3. Painting Strategies to Explore

We will implement three distinct strategies for stroke placement and execution:

| Strategy | Logic | Visual Result |
| --- | --- | --- |
| **A: Superpixel Seeding** | Divide image into SLIC clusters; sample the centroid of each cluster. | Organized, "tiled" look; very clean boundaries. |
| **B: Error-Driven** | Compare the current canvas to the source image; place strokes where the difference is highest. | Focuses detail on complex areas (eyes, edges) while leaving backgrounds broad. |
| **C: Particle Flow** | Treat the image as a fluid field. "Drop" particles and let them drift along color contours. | Long, flowing, impressionistic strokes. |

---

## 4. Technical Specifications & "The Levers"

The script must allow the user to tune these variables to achieve different styles (e.g., Impressionism vs. Pointillism):

### Path Logic

* **Step Size:** The distance between "stamps" within a single continuous stroke.
* **Curvature Penalty:** How strictly the brush follows a curving edge before stopping.
* **Maximum Stroke Length:** Prevents a single stroke from traveling across the entire image.

### Coverage & Termination

* **Coverage Threshold:** % of the canvas that must be non-white before the script stops (e.g., 98%).
* **Distance Transform:** Using `cv2.distanceTransform` to find the "emptiest" spots on the canvas for the next stroke seed.

---

## 5. Proposed Workflow (The Logic Loop)

1. **Preprocessing:** Generate a **Gradient Map** and a **Saliency Map** (to find areas of interest).
2. **Initial Block-in:** Use a very large version of the brush to fill the background quickly.
3. **Refinement Loop:**
* Find the coordinate  with the highest error or lowest coverage.
* Determine the local angle .
* Initiate a stroke: Step forward in direction .
* Update the **Coverage Mask**.


4. **Detail Pass:** Use a small brush size to paint high-contrast edges.

---

## 6. Success Metrics

* **Geometry Preservation:** Do the strokes follow the "flow" of the subject (e.g., circular strokes for a ball)?
* **Texture Fidelity:** Is the brush texture visible, or is it blurred?
* **Efficiency:** Can the script process a 1080p image in under 60 seconds?

