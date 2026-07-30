"""Дискретно-событийная модель пропускной способности участка (SimPy) — быстрая
аналитическая оценка отдельно от полной физики PyBullet, которая для 8 часов
модельного времени по расчётному бюджету не годится (полный прогон в PyBullet
идёт в реальном времени физического шага, а не ускоренно).

Бюджет задержки контура (детекция -> классификация -> подъезд к шиберу -> ход
шибера) содержательно продолжает более ранний числовой расчёт команды (черновики
main.py/pak_simulation.py, до перехода на шиберную PyBullet-симуляцию они были
удалены из репозитория) — но здесь это честная очередь SimPy с ресурсами
(накопитель конечной ёмкости, шиберы как ресурсы с capacity=1, конфликтующие за
доступ), а не отдельный подгонянный numpy-скрипт.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import simpy

from core.types import Config


@dataclass
class ThroughputResult:
    arrival_rate_items_per_hour: float
    items_generated: int
    items_processed: int
    overflow_events: int
    accumulator_capacity: int
    max_accumulator_occupancy: int
    cycle_times_s: list[float] = field(default_factory=list)
    gate_utilization: dict[str, float] = field(default_factory=dict)

    @property
    def cycle_time_p50(self) -> float | None:
        return float(np.median(self.cycle_times_s)) if self.cycle_times_s else None

    @property
    def cycle_time_p99(self) -> float | None:
        return float(np.percentile(self.cycle_times_s, 99)) if self.cycle_times_s else None

    @property
    def overflow_rate(self) -> float:
        return self.overflow_events / self.items_generated if self.items_generated else 0.0


def run_throughput_model(config: Config, arrival_rate_items_per_hour: float,
                          accumulator_capacity: int = 5,
                          category_mix: dict[str, float] | None = None,
                          duration_s: float | None = None,
                          seed: int | None = None) -> ThroughputResult:
    rng = np.random.default_rng(seed if seed is not None else config.seed)
    category_mix = category_mix or {"B": 0.5, "C": 0.2, "D": 0.3}
    categories = list(category_mix.keys())
    probs = list(category_mix.values())
    duration_s = duration_s if duration_s is not None else config.duration_s

    env = simpy.Environment()
    accumulator = simpy.Resource(env, capacity=accumulator_capacity)
    gates = {"C": simpy.Resource(env, capacity=1), "D": simpy.Resource(env, capacity=1)}
    stroke_s = (config.belt_width_mm + 450.0) / 1000.0 / config.gate_max_velocity_mps
    gate_cycle_s = stroke_s * 2.0  # выброс + возврат, см. sim/gates.py Gate.fire

    stats = {
        "generated": 0, "processed": 0, "overflow": 0,
        "cycle_times": [], "accumulator_occupancy": [],
        "gate_busy_time": {"C": 0.0, "D": 0.0},
    }
    mean_interval_s = 3600.0 / arrival_rate_items_per_hour

    def item_process(category: str):
        t0 = env.now
        if accumulator.count >= accumulator_capacity:
            stats["overflow"] += 1
        with accumulator.request() as req:
            yield req
            stats["accumulator_occupancy"].append(accumulator.count)

            # накопление кадра + вычисление классификации — те же порядки величин,
            # что таймауты DETECT/CLASSIFY в core/fsm.py на физическом контуре
            yield env.timeout(max(rng.normal(0.05, 0.01), 0.005))
            yield env.timeout(max(rng.normal(0.01, 0.002), 0.001))

            if category == "B":
                transit_s = config.a_to_b_long_offset_mm / 1000.0 / config.belt_speed_mps
                yield env.timeout(transit_s)
                stats["processed"] += 1
                stats["cycle_times"].append(env.now - t0)
                return

            travel_s = config.gate_distance_mm[category] / 1000.0 / config.belt_speed_mps
            yield env.timeout(travel_s)

            gate = gates[category]
            with gate.request() as greq:
                yield greq
                yield env.timeout(gate_cycle_s)
                stats["gate_busy_time"][category] += gate_cycle_s

            stats["processed"] += 1
            stats["cycle_times"].append(env.now - t0)

    def generator():
        while True:
            yield env.timeout(rng.exponential(mean_interval_s))
            category = str(rng.choice(categories, p=probs))
            stats["generated"] += 1
            env.process(item_process(category))

    env.process(generator())
    env.run(until=duration_s)

    return ThroughputResult(
        arrival_rate_items_per_hour=arrival_rate_items_per_hour,
        items_generated=stats["generated"],
        items_processed=stats["processed"],
        overflow_events=stats["overflow"],
        accumulator_capacity=accumulator_capacity,
        max_accumulator_occupancy=max(stats["accumulator_occupancy"], default=0),
        cycle_times_s=stats["cycle_times"],
        gate_utilization={z: stats["gate_busy_time"][z] / duration_s for z in ("C", "D")},
    )


def find_overflow_threshold(config: Config, rates_per_hour: list[float],
                             accumulator_capacity: int = 5,
                             overflow_rate_limit: float = 0.001,
                             duration_s: float = 600.0,
                             seed: int | None = None) -> list[ThroughputResult]:
    """Прогоняет модель на растущей последовательности темпов подачи (rates_per_hour,
    по возрастанию) и возвращает результаты по каждому — вызывающий код (run.py)
    находит первый темп, где overflow_rate превышает overflow_rate_limit."""
    results = []
    for rate in rates_per_hour:
        results.append(run_throughput_model(
            config, rate, accumulator_capacity=accumulator_capacity,
            duration_s=duration_s, seed=seed))
    return results
