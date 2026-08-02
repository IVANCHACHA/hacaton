"""Real image-processing pipeline for the Gazebo RGB cameras
(worlds/pak_line.sdf: `camera_top/image`, `camera_side/image`).

No ROS2 import here (takes plain numpy uint8 arrays, not Image messages),
so it is unit-testable without rclpy - `perception_node.py` does the
message unwrapping and calls into this module.

History / why RGB, not depth
-----------------------------
This used to be a depth-camera pipeline (foreground = anything closer than
a known static background distance). It was rewritten around the
appearance/contour-based approach from https://github.com/irwkc/robozone -
a working real-camera classifier for this exact track (iPhone/webcam feed,
same B/C/D rule, same K=r_inscribed/R_circumscribed>=0.80 threshold). Two
things carry over directly because they are camera-agnostic:

  - the circle test itself (`formal_k` below is a near-literal port of that
    repo's `classify.formal_k` - inscribed/circumscribed circle radii on
    the filled silhouette, nothing camera-specific about it);
  - "capture a real empty-scene background once, then segment by
    difference from it" (this project's own `accumulate_background`
    already existed for the depth version - see docs/GAZEBO.md bug #6 -
    and generalises to plain RGB colour distance without much change).

What did NOT carry over: robozone's segmentation heuristics
(_appearance_mask/_belt_background/MOG2 motion tracking/PassageDecider
multi-frame voting/ruler-based mm-per-pixel calibration/FastSAM
refinement). Those exist there to handle a live, continuously-moving
handheld/webcam feed with real shadows, glare and unknown camera-to-object
distance - none of which apply here: our cameras are fixed at a known
pose, gz_bridge_node already tells us exactly when to look
(`/pak/sensor/trigger`), and Gazebo's render has no real-world lighting
noise to fight. So segmentation here is the much simpler "diff from a
calibrated background frame" - see `segment()`.

Why a simulated camera doesn't need robozone's ruler calibration
-------------------------------------------------------------------
A handheld/webcam RGB camera doesn't know its own distance to the object,
so robozone recovers mm-per-pixel from a physical ruler visible in frame
each session. Our cameras are static, at a known pose relative to the
belt (see worlds/pak_line.sdf comments) - mm-per-pixel is just
`distance_to_item_plane / fx`, a fixed constant per camera, no in-frame
reference object needed. This is also *why* the top camera alone cannot
measure height: a single RGB view has no depth channel at all (the old
depth-camera version got height "for free" from the closest depth
reading), so height_mm now comes from the side camera's silhouette
bounding-box height instead (see measure_side, and robozone's
side_view.py which does the same thing for the same reason).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    horizontal_fov: float
    item_distance_m: float   # known camera-to-item-plane distance (fixed pose, see worlds/pak_line.sdf)
    diff_threshold: float = 28.0   # per-pixel RGB distance to count as foreground
    min_px: int = 80               # ignore specks/noise (contour area in px)

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / np.tan(self.horizontal_fov / 2.0)

    @property
    def mm_per_px(self) -> float:
        return (self.item_distance_m / self.fx) * 1000.0


# Must stay in sync with worlds/pak_line.sdf (see comments there).
TOP_CAMERA = CameraSpec(name="top", width=320, height=240, horizontal_fov=1.0,
                         item_distance_m=1.2 - 0.5)     # cam_z=1.2, belt_top_z=0.5
SIDE_CAMERA = CameraSpec(name="side", width=320, height=240, horizontal_fov=1.0,
                          item_distance_m=0.6 - 0.0)    # cam_y=0.6, belt centreline y=0


def decode_rgb(data: bytes, width: int, height: int) -> np.ndarray:
    """sensor_msgs/Image with encoding 'rgb8' -> HxWx3 uint8 array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != width * height * 3:
        raise ValueError(f"rgb buffer size {arr.size} != {width}x{height}x3")
    return arr.reshape(height, width, 3).copy()


def accumulate_background(current: np.ndarray | None, frame: np.ndarray,
                           n_prev: int = 0) -> np.ndarray:
    """Equal-weight running-average float32 background, used to build a
    `segment()` calibration reference from a few startup frames before any
    item has entered view (see perception_node.py CALIBRATION_FRAMES).

    `n_prev` is how many frames have already been folded into `current`
    (0 on the first call). The previous `(current + f) / 2` form was NOT a
    true mean - it exponentially favoured the latest frame - so the last
    calibration frame dominated the reference. Equal weighting keeps the
    empty-belt average stable across the whole calibration window.
    """
    f = frame.astype(np.float32)
    if current is None or n_prev <= 0:
        return f
    return (current * float(n_prev) + f) / float(n_prev + 1)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Ring/washer-shaped items must not lose their inscribed-circle
    radius to their own hole - ported from robozone/classify.py."""
    h, w = mask.shape[:2]
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def segment(frame: np.ndarray, background: np.ndarray, spec: CameraSpec) -> np.ndarray | None:
    """Foreground = per-pixel colour distance from the calibrated
    background frame beyond spec.diff_threshold. Returns the largest
    resulting contour, or None if nothing (or only noise) is in view."""
    diff = frame.astype(np.float32) - background
    dist = np.linalg.norm(diff, axis=2)
    mask = (dist > spec.diff_threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < spec.min_px:
        return None
    return cnt


def circumscribed_circle(contour: np.ndarray) -> float:
    _center, r = cv2.minEnclosingCircle(contour)
    return float(r)


def formal_k(contour: np.ndarray, shape_hw: tuple[int, int]) -> float:
    """K = r_inscribed / R_circumscribed on the filled silhouette - the
    exact circularity test from the task statement (Q&A #73/#113), ported
    near-literally from robozone/classify.py::formal_k. Camera-agnostic:
    operates purely on the 2D contour/mask, doesn't care where it came
    from."""
    if contour is None or len(contour) < 3:
        return 0.0
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = _fill_holes(mask)
    # light close only - a 5x5 CLOSE was rounding rectangle corners enough
    # to push K of a 300x200 box from ~0.55 up past the 0.8 circularity
    # threshold (live: Короб 300х200х200 scored K_top=0.823 → false D).
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    solid = max(cnts, key=cv2.contourArea)
    r_out = circumscribed_circle(solid)
    if r_out < 1.0:
        return 0.0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    r_in = float(dist.max())
    return float(min(1.0, r_in / r_out))


@dataclass
class VisionReading:
    k: float
    n_px: int
    length_mm: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    length_px: float | None = None
    width_px: float | None = None


def top_mm_per_px(height_mm: float = 0.0) -> float:
    """mm-per-pixel for the TOP camera, corrected for the item's own
    height. TOP_CAMERA.item_distance_m is the distance to the *belt*, but
    the camera actually images the item's *top surface*, which sits
    height_mm closer to the camera - measuring a tall item's footprint at
    the belt-distance scale systematically overstates it (observed live:
    Шлем, 281.63mm tall, its true 352mm length measured as ~590mm at the
    uncorrected scale - comfortably over the 450mm gabarit line, a false
    positive that isn't just a rounding error). height_mm should come
    from the side camera's reading (see perception_node._on_trigger,
    which has both by the time it needs this); 0.0 (assume flat/
    belt-height) is only a fallback for when no side reading exists yet."""
    distance_m = max(0.05, TOP_CAMERA.item_distance_m - height_mm / 1000.0)
    return (distance_m / TOP_CAMERA.fx) * 1000.0


def measure_top(frame: np.ndarray, background: np.ndarray) -> VisionReading | None:
    """Footprint (length/width in PIXELS, via minAreaRect on the segmented
    contour) + K_top. Deliberately does not convert to mm here - that
    needs the item's height for the perspective correction above, which
    this single top-down view doesn't have (see top_mm_per_px). No height
    either - a single top-down RGB view has no depth channel at all (see
    module docstring); height_mm comes from measure_side instead."""
    cnt = segment(frame, background, TOP_CAMERA)
    if cnt is None:
        return None
    (_c, (ma, mb), _ang) = cv2.minAreaRect(cnt.astype(np.float32))
    length_px, width_px = max(ma, mb), min(ma, mb)
    k = formal_k(cnt, frame.shape[:2])
    return VisionReading(
        k=k, n_px=int(cv2.contourArea(cnt)), length_px=length_px, width_px=width_px)


def measure_side(frame: np.ndarray, background: np.ndarray) -> VisionReading | None:
    """height_mm (silhouette bounding-box height) + K_side - the second,
    independent cross-section required by the "circle in any of 3
    orthogonal projections" rule (Q&A #22/#24/#27, docs/REPORT.md 3.3)."""
    cnt = segment(frame, background, SIDE_CAMERA)
    if cnt is None:
        return None
    _x, _y, _bw, bh = cv2.boundingRect(cnt)
    k = formal_k(cnt, frame.shape[:2])
    return VisionReading(
        k=k, n_px=int(cv2.contourArea(cnt)),
        height_mm=bh * SIDE_CAMERA.mm_per_px)
