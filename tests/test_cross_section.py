"""K = r_вписанной/R_описанной должен считаться по срезам ГЛАВНЫХ ОСЕЙ ИНЕРЦИИ, а
не по силуэту сверху — иначе лежащий цилиндр (сверху выглядит прямоугольником)
будет ошибочно принят за некруглый."""
from __future__ import annotations

import math

import pytest
import trimesh

from core.cross_section import compute_k


def test_lying_cylinder_cross_section_is_round(synthetic_meshes):
    mesh = synthetic_meshes["cylinder_lying_r60_l300"]
    result = compute_k(mesh, n_sections=12)
    assert result.k_value == pytest.approx(1.0, abs=0.01)
    assert result.r_inscribed_mm == pytest.approx(60.0, abs=1.0)


def test_box_k_matches_most_square_face_diagonal_formula(synthetic_meshes):
    # для прямоугольника a>=b: r=b/2, R=sqrt(a^2+b^2)/2, K=b/sqrt(a^2+b^2).
    # максимум по трём граням короба 300x200x150 даёт грань (200,150).
    mesh = synthetic_meshes["box_300x200x150"]
    result = compute_k(mesh, n_sections=12)
    expected = 150.0 / math.sqrt(200.0 ** 2 + 150.0 ** 2)
    assert result.k_value == pytest.approx(expected, abs=1e-3)


def test_regular_hexagon_prism_k_is_cos30(synthetic_meshes):
    mesh = synthetic_meshes["hexagon_prism_r100_l300"]
    result = compute_k(mesh, n_sections=12)
    assert result.k_value == pytest.approx(math.cos(math.radians(30)), abs=1e-3)


def test_hollow_shell_outer_silhouette_is_round(synthetic_meshes):
    """Регрессия: полая труба/бутылка не должна давать заниженный K из-за того, что
    r посчитан внутри тонкого кольца-стенки вместо наружного силуэта сечения."""
    mesh = synthetic_meshes["hollow_cylinder_shell"]
    result = compute_k(mesh, n_sections=12)
    assert result.k_value == pytest.approx(1.0, abs=0.01)
    assert result.r_inscribed_mm == pytest.approx(60.0, abs=1.0)  # наружный радиус, не (60-55)/2


def test_sphere_k_is_one(synthetic_meshes):
    mesh = synthetic_meshes["sphere_r80"]
    result = compute_k(mesh, n_sections=12)
    assert result.k_value == pytest.approx(1.0, abs=0.02)


def test_rotated_box_k_is_rotation_invariant(synthetic_meshes):
    """K — свойство формы, не должен зависеть от того, как тело повёрнуто в мировых
    координатах (в отличие от силуэта сверху)."""
    axis_aligned = compute_k(trimesh.creation.box(extents=[250, 180, 120]), n_sections=12)
    rotated = compute_k(synthetic_meshes["box_rotated"], n_sections=12)
    assert rotated.k_value == pytest.approx(axis_aligned.k_value, abs=1e-3)
