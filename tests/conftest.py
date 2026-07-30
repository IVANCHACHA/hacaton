"""Синтетические тела для юнит-тестов классификатора/K, генерируются один раз в
assets/meshes/synthetic/. Если папка пуста (свежий чекаут), пересобираем на лету —
тесты не должны зависеть от того, что кто-то руками прогнал скрипт-генератор."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

SYNTH_DIR = Path(__file__).resolve().parent.parent / "assets" / "meshes" / "synthetic"


def _build_synthetic_meshes() -> dict[str, trimesh.Trimesh]:
    meshes: dict[str, trimesh.Trimesh] = {}

    meshes["box_300x200x150"] = trimesh.creation.box(extents=[300, 200, 150])

    cyl = trimesh.creation.cylinder(radius=60, height=300, sections=64)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    meshes["cylinder_lying_r60_l300"] = cyl

    meshes["sphere_r80"] = trimesh.creation.icosphere(radius=80, subdivisions=4)

    # полый цилиндр (труба/бутылка со стенкой) — регрессия на баг, найденный на
    # реальном STL организаторов (Бутылка.stl): срез полого тела даёт тонкое кольцо,
    # и если считать r внутри самого кольца, K занижается на порядок. r/R обязаны
    # считаться по НАРУЖНОМУ силуэту сечения, дыры (толщина стенки) роли не играют.
    hollow = trimesh.creation.annulus(r_min=55, r_max=60, height=300)
    hollow.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    meshes["hollow_cylinder_shell"] = hollow
    meshes["capsule_r50_h200"] = trimesh.creation.capsule(radius=50, height=200, count=[32, 32])
    meshes["plate_1_5mm"] = trimesh.creation.box(extents=[200, 150, 1.5])
    meshes["box_oversize_600"] = trimesh.creation.box(extents=[600, 400, 300])

    box_rotated = trimesh.creation.box(extents=[250, 180, 120])
    box_rotated.apply_transform(trimesh.transformations.euler_matrix(0.4, 0.7, 1.1))
    meshes["box_rotated"] = box_rotated

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hexagon_2d = np.stack([100.0 * np.cos(angles), 100.0 * np.sin(angles)], axis=1)
    meshes["hexagon_prism_r100_l300"] = trimesh.creation.extrude_polygon(
        Polygon(hexagon_2d), height=300)

    return meshes


@pytest.fixture(scope="session")
def synthetic_meshes() -> dict[str, trimesh.Trimesh]:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    meshes = _build_synthetic_meshes()
    for name, mesh in meshes.items():
        path = SYNTH_DIR / f"{name}.stl"
        if not path.exists():
            mesh.export(path)
    return meshes
