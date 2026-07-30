"""Правило классификации товара, строго в порядке из ТЗ:
1. Габариты (OBB, не AABB) в диапазоне (min, max) по отсортированным по убыванию размерам.
   Не проходит -> категория C, проверка формы не выполняется.
2. Проходит -> K = r_вписанной/R_описанной (core.cross_section). K > k_threshold -> D.
3. Иначе -> B.
"""
from __future__ import annotations

import trimesh

from core.cross_section import compute_k
from core.types import Category, ClassificationResult, Config


def obb_extents_sorted_mm(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    """Размеры минимального ориентированного бокса (OBB), отсортированы по убыванию."""
    _, extents = trimesh.bounds.oriented_bounds(mesh)
    a, b, c = sorted(extents.tolist(), reverse=True)
    return (a, b, c)


def classify(mesh: trimesh.Trimesh, config: Config) -> ClassificationResult:
    """Классифицирует товар по его 3D-мешу (алгоритм — core.cross_section.compute_k —
    полностью валидирован юнит-тестами на синтетических телах и на реальных STL
    организаторов, см. tests/).

    Осознанное упрощение: здесь classify() получает эталонный меш напрямую, а не
    восстанавливает его из кадра камеры. Технически исполнительный контур (FSM,
    DETECT/CONFIRM) честно работает по RGB+depth backend'а без ground truth (см.
    core/vision.py, core/fsm.py) — но полное восстановление 3D-формы (нужное именно
    для K, не для одних габаритов) из одного взгляда сверху принципиально невозможно:
    лежащий цилиндр сверху и так выглядит прямоугольником, у одной камеры нет данных
    о профиле снизу/сбоку. Индустриальная практика для одних ГАБАРИТОВ — оценка по
    калиброванной камере сверху (см. справочный раздел ТЗ); для K на реальном стенде
    нужен второй ракурс/структурированная подсветка/3D-скан — это отдельная инженерная
    задача за пределами объёма хакатона, и она не меняет сам алгоритм классификации,
    только то, ОТКУДА он берёт меш."""
    extents = obb_extents_sorted_mm(mesh)
    min_sorted = tuple(sorted(config.min_dims_mm, reverse=True))
    max_sorted = tuple(sorted(config.max_dims_mm, reverse=True))

    fits_dims = all(min_sorted[i] < extents[i] < max_sorted[i] for i in range(3))
    if not fits_dims:
        reason = (f"OBB {tuple(round(v, 1) for v in extents)} мм вне диапазона "
                   f"({min_sorted} .. {max_sorted}) мм — не подходит по габаритам")
        return ClassificationResult(Category.C, extents, None, reason)

    cs = compute_k(mesh, n_sections=config.cross_section_slices_per_axis)
    if cs.k_value > config.k_threshold:
        reason = (f"K={cs.k_value:.3f} > порог {config.k_threshold} "
                   f"(ось {cs.axis_index}, r={cs.r_inscribed_mm:.1f}, R={cs.r_circumscribed_mm:.1f}) "
                   f"— круг в сечении, требует доупаковки")
        return ClassificationResult(Category.D, extents, cs.k_value, reason)

    reason = f"K={cs.k_value:.3f} <= порог {config.k_threshold}, габариты в норме — подходит для сортировки"
    return ClassificationResult(Category.B, extents, cs.k_value, reason)
