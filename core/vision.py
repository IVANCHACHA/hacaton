"""Лёгкая обработка кадров backend'а. Работает по RGB+depth, БЕЗ сегментационной
маски (та доступна в sim только для ground truth оценки, см. sim/cameras.py)."""
from __future__ import annotations

import numpy as np

from core.backend import Frame


def area_occupied(frame: Frame, reference: Frame, tolerance_mm: float = 30.0,
                   min_fraction: float = 0.05) -> bool:
    """Есть ли что-то новое в зоне наблюдения камеры: попиксельное фоновое вычитание
    текущего кадра относительно reference (кадр пустой сцены, см. Backend.reference_frame) —
    не константа "пустой фон на X мм", которая легко расходится с реальной геометрией
    (борта, лоток, что угодно ещё в кадре). Считаем занятым, если ощутимая ДОЛЯ кадра
    (min_fraction) ближе камеры, чем фон минус допуск — товар не обязан попадать точно
    в центр кадра, чтобы быть замеченным. Используется и для DETECT (что-то появилось
    в зоне детекции), и для CONFIRM (товар физически долетел до лотка отвода)."""
    depth = frame.depth_mm
    ref = reference.depth_mm
    valid = np.isfinite(depth) & np.isfinite(ref) & (depth > 0) & (ref > 0)
    if not np.any(valid):
        return False
    closer = (ref[valid] - depth[valid]) > tolerance_mm
    return float(np.mean(closer)) >= min_fraction
