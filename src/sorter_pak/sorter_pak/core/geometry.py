"""Geometry primitives for Track 3 (Ozon Robozon) item classification.

Implements the formal rules from the task statement ("Постановка_Задача_3"):

  - gabarit check against the main sorter limits: min 10x10x10 mm,
    max 450x320x320 mm (checked on the item's *true* 3D dimensions,
    independent of how it happens to be lying on the belt - see Q&A #1).
  - "circle in cross-section" check: K = r_inscribed / R_circumscribed,
    evaluated on three mutually orthogonal 2D projections of the object;
    the object is treated as circular if K >= K_THRESHOLD in ANY of the
    three projections (Q&A #6, #14, #22, #24, #27, #73, #113).

No ROS2 / rclpy imports here on purpose: this module is reused both by
the ROS2 nodes and by the standalone CLI tools / web demo generator so
it can be exercised without a ROS2 install.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Task constants (see тз/Постановка_Задача_3_сжато_2.pdf, раздел 2, и Q&A #73)
# ---------------------------------------------------------------------------
GABARIT_MIN_MM = (10.0, 10.0, 10.0)
GABARIT_MAX_MM = (450.0, 320.0, 320.0)
K_THRESHOLD = 0.8
PAK_INPUT_MAX_MM = (500.0, 500.0, 500.0)   # max size accepted onto belt A (Q68/Q114)
PAK_MAX_WEIGHT_KG = 20.0                    # Q4 / Q12 / Q47


# ---------------------------------------------------------------------------
# 2D computational geometry: minimum enclosing circle & maximum inscribed
# circle of a convex polygon (used to compute K = r_in / R_out).
# ---------------------------------------------------------------------------

def _circle_from_two(a: np.ndarray, b: np.ndarray):
    center = (a + b) / 2.0
    r = np.linalg.norm(a - b) / 2.0
    return center, r


def _circle_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        # (near-)collinear points: fall back to the circle of the two
        # farthest-apart points, which bounds all three in this case.
        pairs = [(a, b), (a, c), (b, c)]
        far = max(pairs, key=lambda p: np.linalg.norm(p[0] - p[1]))
        return _circle_from_two(*far)
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    center = np.array([ux, uy])
    r = np.linalg.norm(center - a)
    return center, r


def _in_circle(center: np.ndarray, r: float, p: np.ndarray, eps: float = 1e-7) -> bool:
    return np.linalg.norm(p - center) <= r + eps


def min_enclosing_circle(points: np.ndarray, seed: int = 0):
    """Smallest enclosing circle of a 2D point set (Welzl, iterative form).

    Runs in expected O(n) time on randomly shuffled input; implemented with
    three nested loops instead of recursion so it never hits Python's
    recursion limit on large point sets (e.g. dense mesh silhouettes).
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    rng = random.Random(seed)
    rng.shuffle(pts)
    n = len(pts)
    if n == 0:
        return np.zeros(2), 0.0
    if n == 1:
        return pts[0], 0.0

    center, r = _circle_from_two(pts[0], pts[1])
    for i in range(2, n):
        if _in_circle(center, r, pts[i]):
            continue
        # pts[i] must be on the boundary of the new circle
        center, r = _circle_from_two(pts[0], pts[i])
        for j in range(1, i):
            if _in_circle(center, r, pts[j]):
                continue
            center, r = _circle_from_two(pts[i], pts[j])
            for k in range(j):
                if _in_circle(center, r, pts[k]):
                    continue
                center, r = _circle_from_three(pts[i], pts[j], pts[k])
    return center, r


def max_inscribed_circle(hull_points: np.ndarray):
    """Chebyshev center of a convex polygon = center/radius of the largest
    circle that fits entirely inside it. Solved as a linear program on the
    polygon's half-plane representation (exact for convex regions)."""
    hull = ConvexHull(hull_points)
    # scipy's hull.equations rows are [A_x, A_y, b] with unit-norm (A_x,A_y)
    # such that A.x + b <= 0 defines the interior.
    eq = hull.equations
    A_ub = np.column_stack([eq[:, 0], eq[:, 1], np.ones(len(eq))])
    b_ub = -eq[:, 2]
    c = [0.0, 0.0, -1.0]  # maximize r  <=>  minimize -r
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                   bounds=[(None, None), (None, None), (0, None)],
                   method="highs")
    if not res.success:
        # degenerate polygon (near zero-area): inscribed radius ~ 0
        centroid = hull_points.mean(axis=0)
        return centroid, 0.0
    cx, cy, r = res.x
    return np.array([cx, cy]), float(r)


def min_area_rect_2d(points2d: np.ndarray):
    """Rotating-calipers minimum-area bounding rectangle of a 2D point set.
    Returns (length, width, angle_rad) - the rectangle's long/short side
    and the world-frame angle of its long side, used by the top camera to
    measure an item's footprint *and* estimate its yaw on the belt."""
    pts = np.asarray(points2d, dtype=float)
    if len(pts) < 3:
        span = pts.max(axis=0) - pts.min(axis=0) if len(pts) else np.zeros(2)
        return float(max(span)), float(min(span)), 0.0
    try:
        hull = ConvexHull(pts)
    except QhullError:
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(max(span)), float(min(span)), 0.0
    hp = pts[hull.vertices]
    n = len(hp)
    best_area = None
    best_dims = (0.0, 0.0)
    best_angle = 0.0
    for i in range(n):
        p1, p2 = hp[i], hp[(i + 1) % n]
        edge = p2 - p1
        norm = np.linalg.norm(edge)
        if norm < 1e-9:
            continue
        ux = edge / norm
        uy = np.array([-ux[1], ux[0]])
        proj_u = hp @ ux
        proj_v = hp @ uy
        w = proj_u.max() - proj_u.min()
        h = proj_v.max() - proj_v.min()
        area = w * h
        if best_area is None or area < best_area:
            best_area = area
            angle = np.arctan2(ux[1], ux[0])
            if w >= h:
                best_dims = (w, h)
                best_angle = angle
            else:
                best_dims = (h, w)
                best_angle = angle + np.pi / 2
    return best_dims[0], best_dims[1], float(best_angle)


def k_factor_for_points(points2d: np.ndarray):
    """Returns (K, r_inscribed, R_circumscribed) for a 2D point cloud,
    using the convex hull of the points (K is a hull-only property)."""
    pts = np.asarray(points2d, dtype=float)
    pts = np.unique(pts.round(6), axis=0)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return 0.0, 0.0, 0.0
    hull_pts = pts[hull.vertices]
    if len(hull_pts) < 3:
        return 0.0, 0.0, 0.0
    _, R_out = min_enclosing_circle(hull_pts)
    _, r_in = max_inscribed_circle(hull_pts)
    if R_out <= 1e-9:
        return 0.0, r_in, R_out
    return r_in / R_out, r_in, R_out


# ---------------------------------------------------------------------------
# 3D: oriented bounding box + the three orthogonal projections used by the
# "круг в сечении" rule.
# ---------------------------------------------------------------------------

def oriented_bbox(mesh):
    """Returns (dims_sorted_desc, to_origin_transform) for a trimesh.Trimesh.

    dims_sorted_desc are the item's true 3 characteristic dimensions
    (largest, mid, smallest), independent of the mesh's arbitrary pose on
    the belt - see Q&A #1 ("ящик 400x400x300 едет боком...")."""
    import trimesh
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    dims_sorted = tuple(sorted(extents, reverse=True))
    return dims_sorted, to_origin


@dataclass
class ProjectionResult:
    name: str
    k: float
    r_in: float
    r_out: float

    @property
    def is_circular(self) -> bool:
        return self.k >= K_THRESHOLD


@dataclass
class ClassificationResult:
    category: str                      # 'B' | 'C' | 'D'
    dims_mm: tuple                     # sorted desc, true object dims
    fits_gabarit: bool
    projections: list = field(default_factory=list)   # list[ProjectionResult]
    reason: str = ""

    @property
    def is_circular(self) -> bool:
        return any(p.is_circular for p in self.projections)


def fits_sorter_gabarit(dims_sorted_desc, gmin=GABARIT_MIN_MM, gmax=GABARIT_MAX_MM) -> bool:
    if any(d < gmin[0] for d in dims_sorted_desc):
        return False
    gmax_sorted = tuple(sorted(gmax, reverse=True))
    return all(d <= g for d, g in zip(dims_sorted_desc, gmax_sorted))


def decide_category(dims_sorted_desc, k_values, k_threshold: float = K_THRESHOLD,
                     low_conf_band: float = 0.03):
    """The single shared B/C/D decision rule (gabarit first, then
    circularity, fail-safe to D on borderline confidence) - used by every
    measurement path in the project (synthetic 2-sensor model, real
    Gazebo depth cameras) so there is exactly one place that encodes the
    task's classification policy. Returns (category, fits, circular, low_conf)."""
    fits = fits_sorter_gabarit(dims_sorted_desc)
    ks = list(k_values)
    circular = any(k >= k_threshold for k in ks)
    low_conf = any(abs(k - k_threshold) <= low_conf_band for k in ks)
    if not fits:
        category = "C"
    elif circular or low_conf:
        category = "D"
    else:
        category = "B"
    return category, fits, circular, low_conf


def classify_reference(mesh, k_threshold: float = K_THRESHOLD) -> ClassificationResult:
    """Ground-truth classifier: exact geometric method described by the
    organizers (Q&A #6/#14/#27), applied in the object's own principal-axis
    frame so the result does not depend on its yaw on the belt.

    Used to (a) validate the pipeline against the confirmed example
    (Мешок.stl, K=0.817 -> D, Q&A #113) and (b) label the synthetic test
    set used to drive the simulation / accuracy metrics.
    """
    import trimesh
    dims_sorted, to_origin = oriented_bbox(mesh)
    fits = fits_sorter_gabarit(dims_sorted)

    if not fits:
        return ClassificationResult(
            category="C", dims_mm=dims_sorted, fits_gabarit=False,
            reason="gabarit outside [10x10x10, 450x320x320] mm")

    verts_local = trimesh.transform_points(mesh.vertices, to_origin)
    planes = [("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))]
    projections = []
    for name, (i, j) in planes:
        k, r_in, r_out = k_factor_for_points(verts_local[:, [i, j]])
        projections.append(ProjectionResult(name=name, k=k, r_in=r_in, r_out=r_out))

    circular = any(p.k >= k_threshold for p in projections)
    category = "D" if circular else "B"
    reason = ("circular cross-section detected (K>=%.2f in at least one "
              "orthogonal projection)" % k_threshold) if circular else "passes gabarit and shape checks"
    return ClassificationResult(category=category, dims_mm=dims_sorted,
                                 fits_gabarit=True, projections=projections, reason=reason)
