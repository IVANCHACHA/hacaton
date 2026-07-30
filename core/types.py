"""Общие типы данных и конфиг ПАК. Ничего не знает про PyBullet или конкретный backend."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Category(str, Enum):
    """Три взаимоисключающие категории товара, см. правила классификации в ТЗ."""

    B = "B"  # подходит для сортировки
    C = "C"  # не подходит по габаритам
    D = "D"  # не подходит без доупаковки (круг в сечении)


@dataclass(frozen=True)
class ClassificationResult:
    category: Category
    obb_extents_mm: tuple[float, float, float]  # отсортированы по убыванию
    k_value: float | None  # None, если до проверки формы дело не дошло (сразу C)
    reason: str


@dataclass
class Config:
    """Все настраиваемые параметры задачи. Ничего не зашито в коде классификатора/FSM —
    только здесь, с возможностью переопределения из yaml сценария и из CLI."""

    # --- классификация ---
    k_threshold: float = 0.7  # ТЗ противоречит само себе (текст: 0.7, иллюстрация: 0.8) —
    # поэтому это параметр конфига, а не константа в коде классификатора.
    min_dims_mm: tuple[float, float, float] = (10.0, 10.0, 2.0)
    max_dims_mm: tuple[float, float, float] = (450.0, 320.0, 320.0)
    cross_section_slices_per_axis: int = 12  # 10-15 по ТЗ

    # --- геометрия участка (мм), см. схему организаторов ---
    work_zone_mm: tuple[float, float] = (6000.0, 10000.0)
    belt_width_mm: float = 500.0
    belt_top_height_mm: float = 700.0
    a_to_b_long_offset_mm: float = 7000.0
    a_to_b_cross_offset_mm: float = 1500.0
    roll_cage_size_mm: tuple[float, float, float] = (1200.0, 800.0, 800.0)
    max_item_size_mm: float = 500.0
    # расстояние от зоны детекции (камера infeed) до каждого шибера, мм — уточняется
    # в sim/scene.py при построении геометрии, здесь — рабочие значения по умолчанию
    gate_distance_mm: dict[str, float] = field(default_factory=lambda: {"C": 1200.0, "D": 2200.0})
    # core.vision.area_occupied: попиксельное сравнение с reference_frame (не константа)
    chute_occupied_tolerance_mm: float = 30.0
    # Камеры на отводах вынужденно смотрят на весь лоток целиком (запас на снос
    # товара во время хода шибера, см. sim/scene.py) — узкий/маленький товар (вплоть
    # до 10x10x2 мм по ТЗ) занимает доли процента такого широкого кадра. Порог взят
    # низким, чтобы не терять именно такие случаи.
    occupancy_min_fraction: float = 0.0005

    # --- динамика ---
    belt_speed_mps: float = 1.0
    physics_dt_s: float = 1.0 / 240.0
    item_max_mass_kg: float = 5.0  # для сайзинга maxForce шибера (запас x2 от этой массы)
    item_density_kg_m3: float = 250.0  # оценка массы товара по объёму меша (лёгкая тара/картон)
    gate_stroke_time_budget_s: float = 0.25  # используется для сайзинга силы (разгон), не скорости
    gate_safety_factor: float = 2.0
    gate_max_velocity_mps: float = 1.3  # ограничение скорости плиты — иначе бьёт по товару и роняет
    max_gate_retries: int = 2  # сколько повторных импульсов шибера допускается в RECOVERY

    # --- режим прогона ---
    mode: str = "physics"  # "physics" (PyBullet, run_scenario) | "throughput" (SimPy, аналитика)
    accumulator_capacity: int = 5
    category_mix: dict[str, float] = field(default_factory=lambda: {"B": 0.5, "C": 0.2, "D": 0.3})
    throughput_rates_per_hour: list[float] = field(default_factory=list)
    overflow_rate_limit: float = 0.001

    # --- сценарий ---
    scenario_name: str = "typical"
    seed: int = 42
    duration_s: float = 60.0
    spawn_interval_s: float = 2.0
    camera_noise: bool = False
    items: list[dict[str, Any]] = field(default_factory=list)
    # нештатные остановки ленты: [{"at_s": 10.0, "duration_s": 3.0}, ...] — сценарий
    # faults явно противоречит базовому допущению "лента не останавливается" из ТЗ,
    # это и есть цель: показать обнаружение и восстановление, а не идеальную работу
    belt_stop_events: list[dict[str, float]] = field(default_factory=list)

    # --- прочее ---
    headless: bool = True
    backend: str = "sim"
    results_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                continue
            if isinstance(value, list) and key != "items":
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def override_from_args(self, args: argparse.Namespace) -> "Config":
        """Применяет CLI-переопределения поверх значений из yaml. CLI имеет приоритет."""
        for key in ("k_threshold", "seed", "backend", "headless", "scenario_name"):
            value = getattr(args, key, None)
            if value is not None:
                setattr(self, key, value)
        return self
