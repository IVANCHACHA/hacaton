#!/usr/bin/env python3
"""Exports one Gazebo (gz-sim / SDF) model per catalog item, built from the
organizers' real STL meshes:

  src/sorter_pak/models/<slug>/
    model.config
    model.sdf              - link with visual + collision + VelocityControl
    meshes/visual.stl       - simplified full mesh (decimated if heavy)

The visual mesh is exported in the item's *canonical body frame* (X=longest,
Y=mid, Z=shortest, centered at the centroid - see
core/stl_dataset.canonical_frame_transform), i.e. "resting on its largest
face" (Q&A #61) with the model origin at its centroid. That makes the
spawn height trivial to compute: z = belt_top_z + dims_mm[2]/2/1000.

Collision geometry is a plain axis-aligned box sized to dims_mm in that same
canonical frame - NOT a mesh (convex-hull or otherwise). This was a deliberate
downgrade from an earlier convex-hull-mesh collision approach: live testing
on a full run (many items simultaneously in contact near the spawn point,
each with its own convex hull) surfaced real, repeated ODE contact-solver
instability - "Trimesh-trimesh contact hash table bucket overflow" warnings
and, worse, items occasionally getting launched to physically absurd
positions (tens of metres away) after a routine contact. A box is the
cheapest, most numerically stable primitive any physics engine has for
box-vs-box and box-vs-mesh (belt/rails) contact, at the cost of collision
fidelity for items that are far from box-shaped (e.g. Цилиндр) - an
acceptable trade here since nothing in this project's classification logic
depends on collision-mesh precision (that's what the depth cameras +
core/vision.py are for), only on items behaving like well-defined rigid
bodies on a belt.

No ROS2/Gazebo needed to run this - pure trimesh/numpy. Re-run it whenever
тз/Stl changes:

    python3 tools/export_gz_models.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "sorter_pak"))

from sorter_pak.core import stl_dataset as sd    # noqa: E402
from sorter_pak.core import geometry as geo       # noqa: E402

MODELS_DIR = os.path.join(REPO_ROOT, "src", "sorter_pak", "models")

SLUG = {
    "Бутылка": "bottle", "Короб 300х200х200": "box_300x200x200",
    "Короб 400х400х300": "box_400x400x300", "ЛанчБокс": "lunchbox",
    "Мешок": "bag", "Пуфик": "pouf", "Ручка": "handle",
    "Тарелка": "plate", "Цилиндр": "cylinder", "Шлем": "helmet",
    "Моющее средство": "detergent",
}

COLOR = {  # per-item-type tint, NOT per-category: a real vision pipeline has
    # to tell round-vs-not-round and in/out of gabarit from *shape*, so the
    # paint colour must not leak the answer (category) to the camera.
    "bottle": "0.15 0.45 0.65 1", "box_300x200x200": "0.55 0.45 0.30 1",
    "box_400x400x300": "0.35 0.55 0.35 1", "lunchbox": "0.70 0.60 0.20 1",
    "bag": "0.60 0.35 0.20 1", "pouf": "0.50 0.20 0.20 1",
    "handle": "0.20 0.20 0.20 1", "plate": "0.75 0.75 0.75 1",
    "cylinder": "0.30 0.30 0.60 1", "helmet": "0.65 0.15 0.15 1",
    "detergent": "0.20 0.55 0.55 1",
}

MODEL_SDF_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="item_template">
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{mass:.4f}</mass>
        <inertia>
          <ixx>{ixx:.6f}</ixx><iyy>{iyy:.6f}</iyy><izz>{izz:.6f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>{dim_x:.4f} {dim_y:.4f} {dim_z:.4f}</size></box>
        </geometry>
        <surface>
          <friction>
            <ode><mu>0.6</mu><mu2>0.6</mu2></ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh><uri>meshes/visual.stl</uri></mesh>
        </geometry>
        <material>
          <ambient>{color}</ambient>
          <diffuse>{color}</diffuse>
        </material>
      </visual>
    </link>
    <!-- items ride the belt at a constant 1 m/s (Q&A #87/#98 - belt speed
         is fixed and cannot be slowed/stopped). Velocity is NOT set here:
         gz_bridge_node.py publishes it on /model/<name>/cmd_vel right
         after spawn (and again, sideways, for the shiber push) - keeping
         a single code path for "what makes this thing move" instead of
         splitting it between this file and the bridge node. -->
    <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl"/>
  </model>
</sdf>
"""

MODEL_CONFIG_TEMPLATE = """<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    Auto-generated from тз/Stl/{source_file} by tools/export_gz_models.py.
    Reference category (see docs/REPORT.md, section 3): {category}.
  </description>
</model>
"""


MIN_INERTIA = 1e-5   # kg*m^2 floor - see box_inertia docstring


def box_inertia(mass, dims_m):
    """Box inertia tensor diagonal, floored to MIN_INERTIA. Ручка's true
    box-inertia ixx (rotation about its own long axis, ~0.0000004 kg*m^2
    for a thin 148x13x9mm rod at the 0.02kg mass floor) rounds to exactly
    "0.000000" at model.sdf's %.6f precision - gz-sim's physics engine
    rejects a literal zero diagonal entry outright ("Invalid entry for
    (0,0): 0" warning) and the resulting degenerate inertia matrix was
    observed live to make that item's rigid-body integration blow up
    (position several metres off after a routine push - see
    docs/GAZEBO.md). Flooring keeps every model's inertia matrix
    well-conditioned; MIN_INERTIA is negligible next to any real item's
    other axes (>=1e-4 for everything in the catalog) so it doesn't change
    how anything actually behaves."""
    a, b, c = dims_m
    ixx = max(mass * (b ** 2 + c ** 2) / 12.0, MIN_INERTIA)
    iyy = max(mass * (a ** 2 + c ** 2) / 12.0, MIN_INERTIA)
    izz = max(mass * (a ** 2 + b ** 2) / 12.0, MIN_INERTIA)
    return ixx, iyy, izz


MM_TO_M = 0.001


def export_one(mesh, item_type, out_dir):
    import trimesh
    to_origin = trimesh.bounds.oriented_bounds(mesh)[0]
    T = sd.canonical_frame_transform(mesh, to_origin)
    # тз/Stl models are authored in millimetres; SDF/Gazebo physics is SI
    # (metres, kg). Bake the mm->m conversion into the exported mesh so
    # model.sdf's <mesh><uri> geometry needs no separate <scale> tag and
    # inertia/mass (already computed in kg/m below) stay consistent.
    S = np.diag([MM_TO_M, MM_TO_M, MM_TO_M, 1.0])
    T = S @ T

    visual = mesh.copy()
    visual.apply_transform(T)
    if len(visual.faces) > 6000:
        try:
            visual = visual.simplify_quadric_decimation(face_count=4000)
        except Exception:
            pass

    meshes_dir = os.path.join(out_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)
    visual.export(os.path.join(meshes_dir, "visual.stl"))

    slug = SLUG[item_type.name]
    dims_m = np.array(item_type.dims_mm) / 1000.0
    ixx, iyy, izz = box_inertia(max(item_type.weight_kg, 0.02), dims_m)
    sdf = MODEL_SDF_TEMPLATE.format(
        mass=max(item_type.weight_kg, 0.02), ixx=ixx, iyy=iyy, izz=izz,
        dim_x=dims_m[0], dim_y=dims_m[1], dim_z=dims_m[2],
        color=COLOR.get(slug, "0.5 0.5 0.5 1"))
    with open(os.path.join(out_dir, "model.sdf"), "w", encoding="utf-8") as fh:
        fh.write(sdf)

    cfg = MODEL_CONFIG_TEMPLATE.format(name=slug, source_file=item_type.source_file,
                                        category=item_type.category_true)
    with open(os.path.join(out_dir, "model.config"), "w", encoding="utf-8") as fh:
        fh.write(cfg)


def main():
    import trimesh
    catalog = sd.build_catalog()
    os.makedirs(MODELS_DIR, exist_ok=True)
    for item_type in catalog:
        slug = SLUG.get(item_type.name)
        if slug is None:
            print(f"skip {item_type.name!r}: no slug mapping")
            continue
        src_path = os.path.join(sd.DEFAULT_STL_DIR, item_type.source_file)
        mesh = trimesh.load(src_path, force="mesh")
        out_dir = os.path.join(MODELS_DIR, slug)
        export_one(mesh, item_type, out_dir)
        print(f"exported {slug:24s} <- {item_type.source_file} "
              f"(dims={tuple(round(d) for d in item_type.dims_mm)} mm, "
              f"cat={item_type.category_true}, mass={item_type.weight_kg}kg)")

    with open(os.path.join(MODELS_DIR, "slugs.json"), "w", encoding="utf-8") as fh:
        import json
        json.dump({it.name: SLUG[it.name] for it in catalog if it.name in SLUG},
                   fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(SLUG)} models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
