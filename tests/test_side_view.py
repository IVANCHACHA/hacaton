#!/usr/bin/env python3
"""Боковая камера: высота в мм и K по вертикальной проекции."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classify import quad_to_pixels  # noqa: E402
from side_view import DEFAULT_SIDE_ROI_QUAD, SideView  # noqa: E402

W, H = 640, 360
MM_PER_PX = 1.0


def background() -> np.ndarray:
    rng = np.random.default_rng(7)
    bg = np.full((H, W, 3), 70, np.uint8)
    # статичная «сцена» за конвейером + чёрная лента внизу
    for i in range(0, W, 40):
        cv2.rectangle(bg, (i, 40), (i + 20, 300), (95, 90, 88), -1)
    cv2.rectangle(bg, (0, 300), (W, H), (28, 28, 30), -1)
    noise = rng.integers(-4, 5, size=bg.shape, dtype=np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def roi_bottom_y() -> int:
    quad = quad_to_pixels(H, W, DEFAULT_SIDE_ROI_QUAD)
    return int(quad[2][1])


def frame_with(shape: str, cx: int, size: int) -> np.ndarray:
    """Объект стоит на линии ленты — низ силуэта на нижней границе ROI."""
    f = background()
    base = roi_bottom_y()
    color = (215, 225, 230)
    if shape == "circle":
        r = size // 2
        cv2.circle(f, (cx, base - r), r, color, -1)
    elif shape == "box":
        cv2.rectangle(f, (cx - size, base - size // 2), (cx + size, base), color, -1)
    return f


def run(shape: str, size: int) -> tuple[float, float]:
    sv = SideView()
    bg = background()
    for _ in range(20):
        sv.measure(bg, MM_PER_PX, 300.0)
    out = None
    for cx in (200, 260, 320, 380):
        out = sv.measure(frame_with(shape, cx, size), MM_PER_PX, 300.0)
    assert out is not None, f"{shape}: силуэт не найден ({sv.drop})"
    return out.height_mm, out.k_side


def test_circle_side() -> None:
    """Лежащий цилиндр сбоку — круг: K_side высокий, высота = диаметр."""
    height, k = run("circle", 80)
    assert abs(height - 80.0) <= 6.0, f"высота круга {height:.1f} мм"
    assert k >= 0.85, f"K_side круга {k:.3f}"


def test_box_side() -> None:
    """Коробка сбоку — прямоугольник: K_side низкий, высота = толщина."""
    height, k = run("box", 60)
    assert abs(height - 30.0) <= 6.0, f"высота коробки {height:.1f} мм"
    assert k <= 0.65, f"K_side коробки {k:.3f}"


def test_empty_belt() -> None:
    sv = SideView()
    bg = background()
    for _ in range(20):
        sv.measure(bg, MM_PER_PX, 300.0)
    assert sv.measure(bg, MM_PER_PX, 300.0) is None


if __name__ == "__main__":
    for fn in (test_circle_side, test_box_side, test_empty_belt):
        fn()
        print(f"ok  {fn.__name__}")
    for shape, size in (("circle", 80), ("box", 60)):
        h, k = run(shape, size)
        print(f"{shape:>7}: H={h:.1f}мм K_side={k:.3f}")
