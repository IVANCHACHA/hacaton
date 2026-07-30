#!/usr/bin/env python3
"""Интерактивная песочница: окно PyBullet со слайдерами скорости ленты и габаритов
товара + кнопка «Заспавнить». Каждый спавн — честный короб с текущими размерами,
проходящий через core.classifier.classify и весь FSM (DETECT -> CLASSIFY -> TRACK ->
ACTUATE -> CONFIRM) как в обычном прогоне — просто без сценарного yaml, товары и
параметры задаются вручную из окна. Результат каждого товара печатается в консоль.

    python tools/playground.py

Запускайте из обычного терминала (не через фоновый процесс без фокуса окна) —
addUserDebugParameter изредка отдаёт "Failed to read parameter" на самом старте,
если окно ещё не получило фокус; при флаковом чтении кадр просто пропускается.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pybullet as p
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.sim_backend import SimBackend  # noqa: E402
from core.classifier import classify  # noqa: E402
from core.fsm import ItemFSM  # noqa: E402
from core.types import Config  # noqa: E402


def main() -> None:
    config = Config()
    backend = SimBackend(config, gui=True)
    scene = backend.scene

    speed_id = p.addUserDebugParameter("Скорость ленты, м/с", 0.1, 2.0, config.belt_speed_mps)
    size_x_id = p.addUserDebugParameter("Габарит X, мм", 10, 500, 300)
    size_y_id = p.addUserDebugParameter("Габарит Y, мм", 10, 500, 200)
    size_z_id = p.addUserDebugParameter("Габарит Z, мм", 2, 500, 150)
    k_id = p.addUserDebugParameter("k_threshold", 0.5, 0.95, config.k_threshold)
    # кнопка в PyBullet — это слайдер с rangeMin > rangeMax; каждый клик увеличивает
    # читаемое значение на 1, отслеживаем изменение вместо абсолютного значения
    spawn_id = p.addUserDebugParameter("Заспавнить товар", 1, 0, 0)

    last_spawn_value = p.readUserDebugParameter(spawn_id)
    active: list[tuple[int, ItemFSM]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="pak_playground_"))
    counter = 0
    dt = config.physics_dt_s

    print("Окно открыто. Двигайте слайдеры, жмите 'Заспавнить товар'. Ctrl+C — выход.")

    try:
        while True:
            frame_t0 = time.perf_counter()
            try:
                speed = p.readUserDebugParameter(speed_id)
                scene.belt_a.speed_mps = speed
                scene.belt_b.speed_mps = speed
                config.belt_speed_mps = speed
                config.k_threshold = p.readUserDebugParameter(k_id)
                spawn_value = p.readUserDebugParameter(spawn_id)
            except p.error:
                # изредка флакует сразу после старта, пока окно не получило фокус —
                # не роняем песочницу из-за одного сбойного чтения, пробуем следующий кадр
                continue

            if spawn_value != last_spawn_value:
                last_spawn_value = spawn_value
                sx = p.readUserDebugParameter(size_x_id)
                sy = p.readUserDebugParameter(size_y_id)
                sz = p.readUserDebugParameter(size_z_id)
                counter += 1
                mesh_path = tmp_dir / f"item_{counter}.stl"
                trimesh.creation.box(extents=[sx, sy, sz]).export(mesh_path)
                mesh = trimesh.load(mesh_path, force="mesh")
                body_id = scene.spawn_item(str(mesh_path))
                fsm = ItemFSM(f"live-{counter}", mesh_path.name, mesh, config, backend, classify)
                fsm.start()
                active.append((body_id, fsm))
                print(f"[{counter}] спавн {sx:.0f}x{sy:.0f}x{sz:.0f} мм, "
                      f"скорость ленты {speed:.2f} м/с, k_threshold={config.k_threshold:.2f}")

            backend.step(1)
            still_active = []
            for body_id, fsm in active:
                fsm.tick(dt)
                if fsm.finished:
                    log = fsm.log
                    k_str = f"{log.k_value:.3f}" if log.k_value is not None else "—"
                    print(f"      -> {log.item_id}: K={k_str} категория={log.category_pred} "
                          f"зона={log.zone_actual} успех={log.success}"
                          + (f" [{log.notes}]" if log.notes else ""))
                    scene.remove_item(body_id)
                else:
                    still_active.append((body_id, fsm))
            active = still_active

            elapsed = time.perf_counter() - frame_t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nВыход.")
    finally:
        backend.disconnect()


if __name__ == "__main__":
    main()
