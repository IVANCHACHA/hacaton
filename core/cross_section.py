"""Признак «круг в сечении».

Это НЕ силуэт сверху: лежащий цилиндр сверху выглядит прямоугольником, но его
поперечное сечение (перпендикулярно оси цилиндра) — круг. Поэтому меш режется
не по осям XYZ мировой системы, а по главным осям инерции (собственные векторы
тензора инерции относительно центра масс) — для цилиндра одна из них совпадает
с его продольной осью, и срезы перпендикулярно двум ОСТАЛЬНЫМ осям как раз дают
круглые сечения независимо от того, как тело лежит на ленте.

K = r_вписанной / R_описанной берётся как максимум по всем срезам всех трёх осей:
    R — минимальная описанная окружность контура (trimesh.nsphere, точный алгоритм
        минимальной n-сферы, не bounding-box по диагонали);
    r — через полюс недоступности (shapely.ops.polylabel), а не через центр масс
        контура: для несимметричных/угловатых сечений «центр» может лежать вне
        фигуры или давать заниженный r, полюс недоступности — центр наибольшей
        вписанной окружности по определению.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import polylabel


@dataclass(frozen=True)
class CrossSectionK:
    k_value: float
    r_inscribed_mm: float
    r_circumscribed_mm: float
    axis_index: int  # 0/1/2 — какая из трёх главных осей дала максимум (для лога/отладки)
    axis_vector: tuple[float, float, float]


_ZERO_K = CrossSectionK(k_value=0.0, r_inscribed_mm=0.0, r_circumscribed_mm=0.0,
                         axis_index=-1, axis_vector=(0.0, 0.0, 0.0))


def principal_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    """Три единичных взаимно ортогональных главных оси инерции, строки массива (3,3)."""
    inertia = mesh.moment_inertia
    _eigvals, eigvecs = np.linalg.eigh(inertia)
    return eigvecs.T


def _largest_polygon(section) -> Polygon | None:
    """Наибольший по площади КОНТУР сечения, с заполненными внутренними дырами.

    Реальные STL товаров бывают моделями полых оболочек (например, бутылка со
    стенкой в пару мм) — срез такого тела даёт тонкое кольцо (наружный контур почти
    у наружного контура). Нас интересует форма НАРУЖНОГО силуэта (катится ли товар
    по сортировщику), а не толщина стенки, поэтому дыры (внутренние контуры) сразу
    отбрасываются и площадь считается по внешнему контуру, а не по площади кольца —
    иначе тонкое кольцо «побеждает» по критерию наибольшей площади над случайным
    маленьким сплошным обломком, и наоборот, r внутри кольца получается на порядок
    меньше настоящего радиуса тела."""
    if section is None:
        return None
    try:
        planar, _ = section.to_planar()
    except Exception:
        return None
    polygons = planar.polygons_full
    if not polygons:
        return None
    exteriors = [Polygon(p.exterior) for p in polygons if p.exterior is not None]
    if not exteriors:
        return None
    return max(exteriors, key=lambda p: p.area)


def _circle_ratio(polygon: Polygon) -> tuple[float, float, float] | None:
    if polygon is None or polygon.area <= 0 or not polygon.exterior:
        return None
    coords = np.array(polygon.exterior.coords)
    if len(coords) < 3:
        return None
    _, R = trimesh.nsphere.minimum_nsphere(coords)
    if R <= 1e-9:
        return None
    pole = polylabel(polygon, tolerance=max(R / 200.0, 1e-3))
    r = polygon.boundary.distance(pole)
    return r, R, r / R


def compute_k(mesh: trimesh.Trimesh, n_sections: int = 12, margin: float = 0.06) -> CrossSectionK:
    """Максимум K по n_sections срезам (10-15 по ТЗ) каждой из трёх главных осей.

    margin отсекает крайние `margin`/`1-margin` доли протяжённости меша вдоль оси,
    где сечение вырождается (точка/грань на конце тела) и K не имеет физического смысла.
    """
    axes = principal_axes(mesh)
    vertices = mesh.vertices
    best = _ZERO_K
    fracs = np.linspace(margin, 1.0 - margin, n_sections)

    for axis_idx in range(3):
        axis = axes[axis_idx]
        t = vertices @ axis
        t_min, t_max = float(t.min()), float(t.max())
        if t_max - t_min < 1e-6:
            continue
        for frac in fracs:
            t_s = t_min + frac * (t_max - t_min)
            plane_origin = axis * t_s
            try:
                section = mesh.section(plane_origin=plane_origin, plane_normal=axis)
            except Exception:
                continue
            polygon = _largest_polygon(section)
            ratio = _circle_ratio(polygon)
            if ratio is None:
                continue
            r, R, k = ratio
            if k > best.k_value:
                best = CrossSectionK(k_value=k, r_inscribed_mm=r, r_circumscribed_mm=R,
                                      axis_index=axis_idx, axis_vector=tuple(axis.tolist()))
    return best
