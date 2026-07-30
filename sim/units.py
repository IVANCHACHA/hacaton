"""Единая точка конвертации мм <-> м. Bullet работает в метрах, весь остальной код
(ТЗ, конфиг, логи) — в миллиметрах. Больше нигде в коде не должно быть /1000 или *1000
для геометрии — только через эти функции."""
from __future__ import annotations

MM_PER_M = 1000.0


def to_m(value_mm: float) -> float:
    return value_mm / MM_PER_M


def to_mm(value_m: float) -> float:
    return value_m * MM_PER_M


def vec_to_m(vec_mm) -> tuple[float, ...]:
    return tuple(v / MM_PER_M for v in vec_mm)


def vec_to_mm(vec_m) -> tuple[float, ...]:
    return tuple(v * MM_PER_M for v in vec_m)
