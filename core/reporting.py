"""results.csv (по товару) и summary.json (агрегаты) — формат из постановки задачи."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from core.fsm import ItemLog

CSV_FIELDS = [
    "item_id", "source_file", "obb_x", "obb_y", "obb_z", "k_value",
    "category_pred", "category_true", "zone_actual",
    "t_detect", "t_classify", "t_pick", "t_place", "t_cycle",
    "grip_attempts", "success", "notes",
]


def write_results_csv(logs: list[ItemLog], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for log in logs:
            obb = log.obb_mm or (None, None, None)
            writer.writerow({
                "item_id": log.item_id,
                "source_file": log.source_file,
                "obb_x": obb[0], "obb_y": obb[1], "obb_z": obb[2],
                "k_value": log.k_value,
                "category_pred": log.category_pred,
                "category_true": log.category_true,
                "zone_actual": log.zone_actual,
                "t_detect": log.t_detect, "t_classify": log.t_classify,
                # t_pick/t_place — из постановки задачи пользователя; для шиберной
                # исполнительной части ближайший смысловой аналог — t_actuate/t_confirm
                "t_pick": log.t_actuate, "t_place": log.t_confirm,
                "t_cycle": log.t_cycle,
                "grip_attempts": log.grip_attempts,
                "success": log.success,
                "notes": log.notes,
            })


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def build_summary(logs: list[ItemLog], duration_s: float) -> dict:
    categories = ["B", "C", "D"]
    confusion = {t: {p: 0 for p in categories + ["NONE"]} for t in categories + ["NONE"]}
    for log in logs:
        true = log.category_true or "NONE"
        pred = log.category_pred or "NONE"
        confusion.setdefault(true, {p: 0 for p in categories + ["NONE"]})
        confusion[true][pred] = confusion[true].get(pred, 0) + 1

    per_category_accuracy = {}
    for cat in categories:
        total = sum(confusion.get(cat, {}).values())
        correct = confusion.get(cat, {}).get(cat, 0)
        per_category_accuracy[cat] = (correct / total) if total else None

    n_with_truth = sum(sum(row.values()) for k, row in confusion.items() if k != "NONE")
    n_correct = sum(confusion.get(c, {}).get(c, 0) for c in categories)
    overall_accuracy = (n_correct / n_with_truth) if n_with_truth else None

    cycle_times = [log.t_cycle for log in logs if log.t_cycle is not None]
    successes = [log for log in logs if log.success]
    faults = [log for log in logs if log.grip_attempts > 1 or (not log.success)]

    return {
        "n_items": len(logs),
        "confusion_matrix": confusion,
        "accuracy_per_category": per_category_accuracy,
        "accuracy_overall": overall_accuracy,
        "cycle_time_s": {
            "mean": (sum(cycle_times) / len(cycle_times)) if cycle_times else None,
            "median": median(cycle_times) if cycle_times else None,
            "p95": _percentile(cycle_times, 0.95),
        },
        "throughput_items_per_hour": (len(logs) / duration_s * 3600.0) if duration_s > 0 else None,
        "success_rate": (len(successes) / len(logs)) if logs else None,
        "fault_events": len(faults),
    }


def write_summary_json(logs: list[ItemLog], path: str | Path, duration_s: float) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(logs, duration_s)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
