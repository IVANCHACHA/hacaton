"""Builds a lightweight, reusable catalog from the organizers' STL set
(тз/Stl/*.stl) so the simulator, the ROS2 nodes and the browser demo don't
each have to re-load full meshes (some have 20-35k vertices).

For every item type we cache:
  - the reference classification (ground-truth category + K per plane),
  - a small 3D convex-hull point cloud in a canonical body frame
    (X = item's longest axis, Y = mid axis, Z = shortest axis - i.e. the
    item resting on its largest/most stable face, see Q&A #61),
  - a plausible mass estimate (no mass data is given for the STL set;
    Q&A #47 only fixes the fleet-wide max of 20 kg, so this is a
    documented assumption used solely for the actuator/force discussion
    in the executive part, not for classification).

Everything here is pure numpy/trimesh - no ROS2 dependency.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from . import geometry as geo

def _find_workspace_root(start: str) -> str:
    """Walk up from `start` until a directory containing тз/Stl is found.
    A fixed `../../../..` hop count only reaches the workspace root when
    running from the source tree (sorter_pak/core/ -> up 4). Under `colcon
    build --symlink-install`, __file__ for this module resolves to the
    *install* tree (install/sorter_pak/lib/python3.X/site-packages/
    sorter_pak/core/ - symlink-install symlinks individual files, and
    __file__ reflects the path a module was loaded from, not the symlink
    target), which needs 7 hops to reach the same workspace root - a fixed
    count silently landed inside install/sorter_pak/lib/ instead. That was
    only ever masked by a stray catalog file already sitting at that wrong
    path; the first run to actually exercise the fallback (no cached file
    yet) found 0 STL files there, cached an empty catalog, and crashed
    gz_bridge_node (ValueError: high <= 0 from an empty rng.integers
    range) - see docs/GAZEBO.md. Walking up dynamically is correct for
    both layouts without needing to special-case either one."""
    d = os.path.abspath(start)
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "тз", "Stl")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(start, "..", "..", "..", ".."))  # last-resort guess


_WORKSPACE_ROOT = _find_workspace_root(os.path.dirname(__file__))
DEFAULT_STL_DIR = os.path.join(_WORKSPACE_ROOT, "тз", "Stl")
DEFAULT_CATALOG_PATH = os.path.join(_WORKSPACE_ROOT, "sim_output", "item_catalog.json")

# rough density guess per item (kg per litre) purely for the executive-part
# force/torque discussion - NOT used anywhere in classification.
_DENSITY_HINTS = {
    "бутылка": 1.0, "моющее": 1.0, "мешок": 0.35, "тарелка": 1.4,
    "шлем": 0.25, "пуфик": 0.15, "цилиндр": 2.7, "короб": 0.12,
    "ланчбокс": 0.3, "ручка": 0.9,
}


def _guess_weight_kg(name: str, dims_mm) -> float:
    key = next((k for k in _DENSITY_HINTS if k in name.lower()), None)
    density = _DENSITY_HINTS.get(key, 0.3)
    volume_l = (dims_mm[0] * dims_mm[1] * dims_mm[2]) / 1e6  # mm^3 -> litres
    fill_factor = 0.5  # objects are rarely solid / fully filled
    return round(min(geo.PAK_MAX_WEIGHT_KG, max(0.02, volume_l * density * fill_factor)), 3)


@dataclass
class ItemType:
    name: str
    source_file: str
    dims_mm: tuple                 # sorted desc, true dims
    category_true: str             # 'B' | 'C' | 'D' from the reference classifier
    k_values: dict                 # {"XY":..,"XZ":..,"YZ":..}
    weight_kg: float
    hull_local: list               # Nx3 list, canonical body frame (X=long,Y=mid,Z=short)

    def to_json(self):
        d = asdict(self)
        return d


MAX_HULL_POINTS = 250   # cap so per-instance re-projection stays cheap (sim + JS demo)


def canonical_frame_transform(mesh, to_origin) -> np.ndarray:
    """4x4 transform (mesh's own frame -> canonical body frame) where axis
    0 = item's longest dimension, axis 1 = mid, axis 2 = shortest (i.e.
    the item lying on its largest face, Q&A #61), centered at the
    centroid. Built as a proper rotation (not a raw column permutation)
    so it can be safely applied to a full visual mesh (preserves face
    winding/normals) as well as to bare point clouds - used by both
    `_canonical_hull` below and `tools/export_gz_models.py`."""
    import trimesh
    verts_local = trimesh.transform_points(mesh.vertices, to_origin)
    extents = verts_local.max(axis=0) - verts_local.min(axis=0)
    order = np.argsort(-extents)          # longest -> shortest
    R = np.zeros((3, 3))
    for new_axis, old_axis in enumerate(order):
        R[new_axis, old_axis] = 1.0
    if np.linalg.det(R) < 0:              # keep it a proper rotation, no mirroring
        R[2, :] *= -1
    center = verts_local.mean(axis=0)
    T_center = np.eye(4)
    T_center[:3, 3] = -center
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    return T_rot @ T_center @ to_origin


def _canonical_hull(mesh, to_origin) -> np.ndarray:
    """Simplified 3D convex-hull vertices of `mesh` in the canonical body
    frame (see canonical_frame_transform). Decimated to <= MAX_HULL_POINTS
    so that per-instance re-projection at simulation time (arbitrary yaw)
    stays cheap; this is only used for the *simulated camera* measurements,
    not for the reference/ground-truth classification (which runs on the
    full mesh)."""
    import trimesh
    hull_mesh = mesh.convex_hull
    if len(hull_mesh.vertices) > MAX_HULL_POINTS:
        try:
            target_faces = max(4 * MAX_HULL_POINTS, 60)
            hull_mesh = hull_mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception:
            pass
    T = canonical_frame_transform(mesh, to_origin)
    verts_canonical = trimesh.transform_points(hull_mesh.vertices, T)
    hull = geo.ConvexHull(verts_canonical)
    return verts_canonical[hull.vertices]


def build_catalog(stl_dir: str = DEFAULT_STL_DIR) -> list:
    import trimesh
    files = sorted(glob.glob(os.path.join(stl_dir, "*.stl"))) + \
        sorted(glob.glob(os.path.join(stl_dir, "*.STL")))
    catalog = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        mesh = trimesh.load(f, force="mesh")
        result = geo.classify_reference(mesh)
        to_origin = trimesh.bounds.oriented_bounds(mesh)[0]
        hull_local = _canonical_hull(mesh, to_origin)
        item = ItemType(
            name=name,
            source_file=os.path.basename(f),
            dims_mm=tuple(round(d, 2) for d in result.dims_mm),
            category_true=result.category,
            k_values={p.name: round(p.k, 4) for p in result.projections},
            weight_kg=_guess_weight_kg(name, result.dims_mm),
            hull_local=np.round(hull_local, 2).tolist(),
        )
        catalog.append(item)
    return catalog


def save_catalog(catalog: list, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([it.to_json() for it in catalog], fh, ensure_ascii=False, indent=2)


def load_catalog(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [ItemType(**r) for r in raw]


def get_or_build_catalog(path: str = None) -> list:
    """Used by the ROS2 nodes (and CLI tools) at startup: load a cached
    catalog if present, otherwise build it from the STL set once and cache
    it. Requires `colcon build --symlink-install` so the installed module
    file still resolves DEFAULT_STL_DIR/DEFAULT_CATALOG_PATH back to the
    repo's тз/Stl and sim_output/ directories (documented in README.md)."""
    path = path or DEFAULT_CATALOG_PATH
    if path and os.path.exists(path):
        cached = load_catalog(path)
        if cached:
            return cached
        # An empty cached catalog is never a legitimate result (there are
        # always 11 STL files) - it means some earlier run's build_catalog()
        # silently found 0 STL files (e.g. DEFAULT_STL_DIR resolved wrong)
        # and cached that empty result. Trusting it blindly crashes every
        # caller downstream (gz_bridge_node's rng.integers(0, 0) -> "high
        # <= 0") for a totally different reason than the real problem.
        # Rebuilding for real gives an honest error if STL files truly
        # aren't reachable, instead of a confusing crash somewhere else.
    cat = build_catalog()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_catalog(cat, path)
    except OSError:
        pass  # read-only install - still usable, just not cached
    return cat


def main_cli():
    """console_scripts entry point: `ros2 run sorter_pak build_catalog [out.json]`."""
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "item_catalog.json"
    cat = build_catalog()
    save_catalog(cat, out)
    print(f"wrote {len(cat)} item types to {out}")
    for it in cat:
        print(f"  {it.name:28s} dims={it.dims_mm} cat={it.category_true} "
              f"K={it.k_values} weight={it.weight_kg}kg")


if __name__ == "__main__":
    main_cli()
