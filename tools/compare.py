#!/usr/bin/env python3
"""Сравнение прогона на реальном стенде и в симуляции — валидация цифрового двойника.

Принимает два results.csv одинакового формата (см. core/reporting.py) и печатает
таблицу расхождений по времени цикла, доле успешных маршрутизаций и точности
габаритов. Данных с реального стенда пока нет (стенд собирается параллельно) —
скрипт готов заранее и проверен на сравнении двух прогонов симуляции между собой.

    python tools/compare.py --rig rig_results.csv --sim results/typical/results.csv
"""
from __future__ import annotations

import argparse
import csv


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: str | None) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict]) -> dict:
    cycle_times = [v for r in rows if (v := _to_float(r.get("t_cycle"))) is not None]
    successes = [r.get("success") in ("True", "true", "1") for r in rows]
    obb_x = [v for r in rows if (v := _to_float(r.get("obb_x"))) is not None]
    obb_y = [v for r in rows if (v := _to_float(r.get("obb_y"))) is not None]
    obb_z = [v for r in rows if (v := _to_float(r.get("obb_z"))) is not None]
    return {
        "n": len(rows),
        "cycle_mean": sum(cycle_times) / len(cycle_times) if cycle_times else None,
        "success_rate": sum(successes) / len(successes) if successes else None,
        "obb_x_mean": sum(obb_x) / len(obb_x) if obb_x else None,
        "obb_y_mean": sum(obb_y) / len(obb_y) if obb_y else None,
        "obb_z_mean": sum(obb_z) / len(obb_z) if obb_z else None,
    }


def _fmt_row(name: str, a: float | None, b: float | None, pct: bool = False) -> str:
    if a is None or b is None:
        return f"{name:32s} {'н/д':>14s} {'н/д':>14s} {'н/д':>14s}"
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    diff = (b - a) * scale
    return (f"{name:32s} {a * scale:12.3f}{suffix:2s} {b * scale:12.3f}{suffix:2s} "
            f"{diff:+12.3f}{suffix:2s}")


def compare(rig_path: str, sim_path: str) -> None:
    rig_rows = load_csv(rig_path)
    sim_rows = load_csv(sim_path)
    rig_s, sim_s = summarize(rig_rows), summarize(sim_rows)

    print(f"Товаров в выборке: стенд(--rig)={rig_s['n']}, симуляция(--sim)={sim_s['n']}")
    if rig_s["n"] == 0 or sim_s["n"] == 0:
        print("Одна из выборок пуста — сравнивать нечего.")
        return

    print(f"\n{'Метрика':32s} {'Стенд':>14s} {'Симуляция':>14s} {'Разница':>14s}")
    print("-" * 76)
    print(_fmt_row("Среднее время цикла, с", rig_s["cycle_mean"], sim_s["cycle_mean"]))
    print(_fmt_row("Доля успешных маршрутизаций", rig_s["success_rate"], sim_s["success_rate"], pct=True))
    print(_fmt_row("Средний OBB X, мм", rig_s["obb_x_mean"], sim_s["obb_x_mean"]))
    print(_fmt_row("Средний OBB Y, мм", rig_s["obb_y_mean"], sim_s["obb_y_mean"]))
    print(_fmt_row("Средний OBB Z, мм", rig_s["obb_z_mean"], sim_s["obb_z_mean"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rig", required=True, help="results.csv с реального стенда")
    parser.add_argument("--sim", required=True, help="results.csv из симуляции")
    args = parser.parse_args()
    compare(args.rig, args.sim)


if __name__ == "__main__":
    main()
