#!/usr/bin/env python3
"""Рендерит видеодемонстрацию сценария: сторонняя обзорная камера, MP4 через OpenCV
(без зависимости от ffmpeg в PATH). Работает через SimBackend напрямую (не Backend
Protocol) — это утилита для видео, а не часть контура классификации/маршрутизации.

    python tools/make_video.py --scenario typical --out typical.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.sim_backend import SimBackend  # noqa: E402
from core.orchestrator import run_scenario  # noqa: E402
from core.types import Config  # noqa: E402
from sim.units import to_m  # noqa: E402

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def make_overview_camera(scene, width=960, height=540):
    center_x = scene.lane_center_x.get("C", 2000.0)
    eye = [to_m(v) for v in (center_x * 0.4, -2200.0, 2600.0)]
    target = [to_m(v) for v in (center_x * 0.6, 0.0, 700.0)]
    view = p.computeViewMatrix(eye, target, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(60, width / height, 0.1, 10.0)
    return view, proj, width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="typical")
    parser.add_argument("--out", default="scenario.mp4")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    scenario_path = SCENARIOS_DIR / f"{args.scenario}.yaml"
    config = Config.from_yaml(scenario_path)
    config.scenario_name = scenario_path.stem

    backend = SimBackend(config, gui=False)
    view, proj, w, h = make_overview_camera(backend.scene)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    frame_every_n_steps = max(int(round(1.0 / config.physics_dt_s / args.fps)), 1)
    step_counter = {"n": 0}

    orig_step = backend.step

    def step_and_record(n: int = 1) -> None:
        for _ in range(n):
            orig_step(1)
            step_counter["n"] += 1
            if step_counter["n"] % frame_every_n_steps == 0:
                _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
                arr = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
                writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    backend.step = step_and_record  # type: ignore[method-assign]

    print(f"Рендер сценария {args.scenario} -> {args.out} ({args.fps} fps)")
    run_scenario(config, backend, verbose=True)

    writer.release()
    backend.disconnect()
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
