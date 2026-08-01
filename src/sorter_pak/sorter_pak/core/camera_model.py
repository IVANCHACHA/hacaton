"""The "as-deployed" 2-sensor perception pipeline.

Design (see REPORT.md, "Часть определения и классификации"):

  - Sensor 1 "TOP"  - overhead RGB camera, orthographic top-down view.
    Yaw-invariant: measures the item's footprint (L, W) via a
    rotating-calipers minimum-area rectangle, and from the same
    rectangle's orientation estimates the item's yaw on the belt.

  - Sensor 2 "PROFILE" - a cross-belt line-scan height profiler (laser /
    ToF line, e.g. Keyence/Gocator-class, or a ToF area camera used in
    line-scan mode - see Q&A #37, which explicitly allows ToF cameras and
    belt-colour changes to support them). Unlike a single side snapshot,
    it is swept along the full length of the item as the belt carries it
    past, so it reconstructs a dense height map z(x, y) over the item's
    whole footprint - i.e. a real (if noisy) partial 3D point cloud, not
    a single 2D silhouette.

Why 2 sensors are enough for the 3-projection rule (Q&A #6/#14/#27):
the reference rule needs 3 orthogonal cross-sections IN THE OBJECT'S OWN
FRAME. A single fixed side camera cannot deliver that on its own (its
projection plane only lines up with the object's true axes for specific,
unpredictable yaw values - this was verified empirically in this project:
a plain top+fixed-side-snapshot pipeline missed a bottle lying along the
belt in ~85% of random yaws). Because sensor 2 is a *scanning* profiler,
not a snapshot, it recovers a genuine dense point cloud; combined with the
yaw estimate from sensor 1, that cloud is rotated back into the item's own
frame and the same 3-projection test as the reference classifier is
applied. Measurement noise (yaw estimation error, profiler quantisation)
is modelled explicitly so residual disagreement with the ground-truth
classifier is quantified rather than assumed away (Q&A #59 asks exactly
for this).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo

# Sensor noise model (documented assumption, no sensor spec given in the
# task - tune here if a concrete camera/profiler datasheet is chosen).
YAW_ESTIMATE_NOISE_DEG = 3.0     # 1-sigma error of the top-camera yaw estimate
POINT_NOISE_MM = 0.4             # 1-sigma per-point noise of the profiler point cloud


def _rotate_z(points_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points_xyz @ rot.T


@dataclass
class CameraReading:
    length_mm: float
    width_mm: float
    height_mm: float
    yaw_true_rad: float
    yaw_est_rad: float
    k_values: dict = field(default_factory=dict)   # reconstructed own-frame K's

    @property
    def dims_sorted(self):
        return tuple(sorted([self.length_mm, self.width_mm, self.height_mm], reverse=True))

    @property
    def yaw_error_deg(self) -> float:
        return float(np.degrees(abs(self.yaw_true_rad - self.yaw_est_rad)))


@dataclass
class MeasuredClassification:
    category: str
    reading: CameraReading
    fits_gabarit: bool
    is_circular: bool
    low_confidence: bool = False


def capture_two_sensors(hull_local: np.ndarray, yaw_rad: float,
                         rng: np.random.Generator | None = None) -> CameraReading:
    """hull_local: Nx3 convex-hull points in canonical body frame
    (X=long, Y=mid, Z=short - item resting on its largest face).
    yaw_rad: true rotation about the vertical (Z) axis as placed on the belt.
    """
    rng = rng or np.random.default_rng()
    hull_local = np.asarray(hull_local, dtype=float)
    world = _rotate_z(hull_local, yaw_rad)

    # --- Sensor 1: TOP camera -> footprint (L, W) + yaw estimate ---------
    top_xy = world[:, [0, 1]]
    length_mm, width_mm, angle_est = geo.min_area_rect_2d(top_xy)
    yaw_noise = np.radians(rng.normal(0.0, YAW_ESTIMATE_NOISE_DEG))
    yaw_est = angle_est + yaw_noise

    # --- Sensor 2: line-scan profiler -> noisy dense point cloud --------
    profiler_cloud = world + rng.normal(0.0, POINT_NOISE_MM, size=world.shape)
    # robust (percentile-trimmed) span: a handful of sparse hull points can
    # otherwise pick up the full tails of the per-point noise and inflate
    # small dimensions (this was verified to matter for near-threshold
    # items, e.g. the ~9 mm-thin "Ручка" handle).
    z_lo, z_hi = np.percentile(profiler_cloud[:, 2], [1, 99])
    height_mm = float(z_hi - z_lo)

    # --- Fusion: undo the estimated yaw to recover the item's own frame -
    reconstructed_local = _rotate_z(profiler_cloud, -yaw_est)

    planes = [("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))]
    k_values = {}
    for name, (i, j) in planes:
        k, _, _ = geo.k_factor_for_points(reconstructed_local[:, [i, j]])
        k_values[name] = k

    return CameraReading(length_mm=length_mm, width_mm=width_mm, height_mm=height_mm,
                          yaw_true_rad=yaw_rad, yaw_est_rad=yaw_est, k_values=k_values)


def classify_from_sensors(hull_local: np.ndarray, yaw_rad: float,
                           k_threshold: float = geo.K_THRESHOLD,
                           low_conf_band: float = 0.03,
                           rng: np.random.Generator | None = None) -> MeasuredClassification:
    reading = capture_two_sensors(hull_local, yaw_rad, rng=rng)
    category, fits, circular, low_conf = geo.decide_category(
        reading.dims_sorted, reading.k_values.values(), k_threshold, low_conf_band)

    return MeasuredClassification(category=category, reading=reading, fits_gabarit=fits,
                                   is_circular=circular, low_confidence=low_conf)


# Backwards-compatible aliases used by earlier drafts / external callers.
capture_two_cameras = capture_two_sensors
classify_from_cameras = classify_from_sensors
