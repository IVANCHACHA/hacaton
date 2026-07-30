"""Многотоварный прогон сценария: спавнит товары по расписанию, ведёт несколько
ItemFSM параллельно (несколько товаров в полёте одновременно — так и моделируются
сценарии вроде «два товара вплотную»), собирает лог по каждому."""
from __future__ import annotations

from pathlib import Path

import trimesh

from backends.sim_backend import SimBackend
from core.classifier import classify
from core.fsm import ItemFSM, ItemLog
from core.types import Config

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "meshes"


def _expand_item_list(config: Config) -> list[dict]:
    expanded: list[dict] = []
    for spec in config.items:
        for _ in range(spec.get("count", 1)):
            expanded.append(spec)
    return expanded


def run_scenario(config: Config, backend: SimBackend, verbose: bool = True) -> list[ItemLog]:
    items_spec = _expand_item_list(config)
    dt = config.physics_dt_s
    active: list[tuple[int, ItemFSM]] = []
    finished_logs: list[ItemLog] = []
    mesh_cache: dict[str, trimesh.Trimesh] = {}

    next_spawn_t = 0.0
    spawn_idx = 0
    t = 0.0
    nominal_speed = config.belt_speed_mps
    stop_events = sorted(config.belt_stop_events, key=lambda e: e["at_s"])
    stop_idx = 0
    belt_stopped = False

    while t < config.duration_s:
        if hasattr(backend, "set_belt_speed"):
            while stop_idx < len(stop_events) and t >= stop_events[stop_idx]["at_s"] and not belt_stopped:
                belt_stopped = True
                stop_end_t = t + stop_events[stop_idx]["duration_s"]
                if verbose:
                    print(f"  t={t:6.2f}s  ЛЕНТА ОСТАНОВЛЕНА (нештатно) до t={stop_end_t:.2f}s")
                backend.set_belt_speed(0.0)
            if belt_stopped and t >= stop_end_t:
                belt_stopped = False
                stop_idx += 1
                if verbose:
                    print(f"  t={t:6.2f}s  лента снова на {nominal_speed} м/с")
                backend.set_belt_speed(nominal_speed)
        if spawn_idx < len(items_spec) and t >= next_spawn_t:
            spec = items_spec[spawn_idx]
            mesh_path = str(ASSETS_DIR / spec["mesh"])
            if mesh_path not in mesh_cache:
                mesh_cache[mesh_path] = trimesh.load(mesh_path, force="mesh")
            body_id = backend.spawn_item(mesh_path)
            item_id = f"item-{spawn_idx:04d}"
            fsm = ItemFSM(item_id, spec["mesh"], mesh_cache[mesh_path], config, backend, classify,
                          category_true=spec.get("category_true"))
            fsm.start()
            active.append((body_id, fsm))
            if verbose:
                print(f"  t={t:6.2f}s  spawn {item_id:10s} ({spec['mesh']})")
            spawn_idx += 1
            next_spawn_t = t + spec.get("interval_s", config.spawn_interval_s)

        backend.step(1)
        t += dt

        still_active = []
        for body_id, fsm in active:
            fsm.tick(dt)
            if fsm.finished:
                finished_logs.append(fsm.log)
                backend.remove_item(body_id)
                if verbose:
                    print(f"  t={t:6.2f}s  done  {fsm.item_id:10s} pred={fsm.log.category_pred} "
                          f"zone={fsm.log.zone_actual} success={fsm.log.success} "
                          f"attempts={fsm.log.grip_attempts}"
                          + (f"  [{fsm.log.notes}]" if fsm.log.notes else ""))
            else:
                still_active.append((body_id, fsm))
        active = still_active

        if spawn_idx >= len(items_spec) and not active:
            break

    for body_id, fsm in active:
        fsm.log.notes = fsm.log.notes or "сценарий завершён по duration_s, товар не долетел до конца"
        finished_logs.append(fsm.log)

    return finished_logs
