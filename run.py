#!/usr/bin/env python3
"""Точка входа ПАК сортировки товаров.

    python run.py --backend sim --scenario typical --headless
    python run.py --backend sim --scenario borderline --gui
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.orchestrator import run_scenario
from core.reporting import write_results_csv, write_summary_json
from core.types import Config

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["sim", "rig"], default="sim")
    parser.add_argument("--scenario", default="typical",
                         help="имя сценария (scenarios/<name>.yaml) или путь к yaml-файлу")
    parser.add_argument("--headless", action="store_true",
                         help="без GUI (это и так поведение по умолчанию, флаг для явности)")
    parser.add_argument("--gui", action="store_true", help="показать окно PyBullet")
    parser.add_argument("--k-threshold", type=float, default=None, dest="k_threshold",
                         help="переопределить порог круглости K из сценария")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--quiet", action="store_true", help="не печатать событие по каждому товару")
    return parser.parse_args()


def resolve_scenario_path(name: str) -> Path:
    direct = Path(name)
    if direct.is_file():
        return direct
    candidate = SCENARIOS_DIR / f"{name}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Сценарий не найден: {name} (искал {direct} и {candidate})")


def run_throughput(config: Config) -> None:
    from throughput.simpy_model import find_overflow_threshold, run_throughput_model

    results_dir = Path(config.results_dir) / config.scenario_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if config.throughput_rates_per_hour:
        print(f"Развёртка темпа подачи: накопитель={config.accumulator_capacity} слотов, "
              f"порог переполнения={config.overflow_rate_limit * 100:.2f}%")
        results = find_overflow_threshold(
            config, list(config.throughput_rates_per_hour),
            accumulator_capacity=config.accumulator_capacity,
            overflow_rate_limit=config.overflow_rate_limit,
            duration_s=config.duration_s, seed=config.seed)
        breaking_rate = None
        for r in results:
            flag = "  <-- переполнение" if r.overflow_rate > config.overflow_rate_limit else ""
            print(f"  {r.arrival_rate_items_per_hour:7.0f} шт/ч: сгенерировано={r.items_generated:5d} "
                  f"обработано={r.items_processed:5d} overflow={r.overflow_events:4d} "
                  f"({r.overflow_rate * 100:5.2f}%) cycle p50={r.cycle_time_p50:.2f}с{flag}")
            if breaking_rate is None and r.overflow_rate > config.overflow_rate_limit:
                breaking_rate = r.arrival_rate_items_per_hour
        summary = {
            "accumulator_capacity": config.accumulator_capacity,
            "overflow_rate_limit": config.overflow_rate_limit,
            "breaking_rate_items_per_hour": breaking_rate,
            "sweep": [
                {"rate": r.arrival_rate_items_per_hour, "generated": r.items_generated,
                 "processed": r.items_processed, "overflow_events": r.overflow_events,
                 "overflow_rate": r.overflow_rate, "cycle_p50_s": r.cycle_time_p50,
                 "cycle_p99_s": r.cycle_time_p99, "gate_utilization": r.gate_utilization}
                for r in results
            ],
        }
        print(f"\nНакопитель начинает переполняться при темпе: "
              f"{breaking_rate if breaking_rate else 'не достигнуто в развёртке'} шт/ч")
    else:
        rate = config.throughput_rates_per_hour[0] if config.throughput_rates_per_hour else 3000.0
        r = run_throughput_model(config, rate, accumulator_capacity=config.accumulator_capacity,
                                  duration_s=config.duration_s, seed=config.seed)
        print(f"Темп подачи: {rate} шт/ч, длительность {config.duration_s / 3600:.2f}ч")
        print(f"Сгенерировано: {r.items_generated}, обработано: {r.items_processed}, "
              f"overflow: {r.overflow_events} ({r.overflow_rate * 100:.3f}%)")
        print(f"Cycle time p50={r.cycle_time_p50:.2f}с p99={r.cycle_time_p99:.2f}с")
        print(f"Загрузка шиберов: C={r.gate_utilization['C'] * 100:.1f}% D={r.gate_utilization['D'] * 100:.1f}%")
        summary = {
            "arrival_rate_items_per_hour": rate, "items_generated": r.items_generated,
            "items_processed": r.items_processed, "overflow_events": r.overflow_events,
            "overflow_rate": r.overflow_rate, "cycle_p50_s": r.cycle_time_p50,
            "cycle_p99_s": r.cycle_time_p99, "gate_utilization": r.gate_utilization,
            "duration_hours": config.duration_s / 3600,
        }

    import json
    json_path = results_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nРезультаты: {json_path}")


def main() -> None:
    args = parse_args()
    scenario_path = resolve_scenario_path(args.scenario)
    config = Config.from_yaml(scenario_path)
    config.scenario_name = scenario_path.stem
    if args.k_threshold is not None:
        config.k_threshold = args.k_threshold
    if args.seed is not None:
        config.seed = args.seed
    if args.results_dir:
        config.results_dir = args.results_dir
    config.backend = args.backend

    print(f"=== ПАК сортировки товаров === backend={config.backend} scenario={config.scenario_name} "
          f"k_threshold={config.k_threshold} seed={config.seed} belt_speed={config.belt_speed_mps} м/с")

    if config.mode == "throughput":
        run_throughput(config)
        return

    if config.backend == "sim":
        from backends.sim_backend import SimBackend
        backend = SimBackend(config, gui=args.gui)
    else:
        from backends.rig_backend import RigBackend
        backend = RigBackend(config)

    logs = run_scenario(config, backend, verbose=not args.quiet)

    results_dir = Path(config.results_dir) / config.scenario_name
    csv_path = results_dir / "results.csv"
    json_path = results_dir / "summary.json"
    write_results_csv(logs, csv_path)
    summary = write_summary_json(logs, json_path, config.duration_s)

    print(f"\n=== Итог: {config.scenario_name} ===")
    print(f"Товаров обработано: {summary['n_items']}")
    if summary["success_rate"] is not None:
        print(f"Успешно промаршрутизировано: {summary['success_rate'] * 100:.1f}%")
    if summary["accuracy_overall"] is not None:
        print(f"Accuracy классификации: {summary['accuracy_overall'] * 100:.1f}%")
    ct = summary["cycle_time_s"]
    if ct["median"] is not None:
        print(f"Cycle time: медиана {ct['median']:.2f}с, p95 {ct['p95']:.2f}с")
    if summary["throughput_items_per_hour"] is not None:
        print(f"Пропускная способность: {summary['throughput_items_per_hour']:.0f} шт/ч")
    print(f"Нештатных событий: {summary['fault_events']}")
    print(f"\nРезультаты: {csv_path}\n            {json_path}")

    if hasattr(backend, "disconnect"):
        backend.disconnect()


if __name__ == "__main__":
    main()
