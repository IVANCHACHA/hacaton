#!/usr/bin/env python3
"""Проверка правил классификации по постановке (трек 3).

B — все три габарита в (10, 450×320×320) и ни в одной проекции нет круга.
C — габариты вне диапазона.
D — K = r_вп/R_оп >= 0.8 хотя бы в одной проекции.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classify import (  # noqa: E402
    CIRCLE_RATIO_THRESHOLD,
    Category,
    Detection,
    PassageDecider,
    classify,
    effective_k,
    formal_k,
    rect_fill_score,
)

SHAPE = (400, 400)


def _contour(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)


def _blank() -> np.ndarray:
    return np.zeros(SHAPE, np.uint8)


def circle_mask(r: int = 140) -> np.ndarray:
    m = _blank()
    cv2.circle(m, (200, 200), r, 255, -1)
    return m


def square_mask(a: int = 240) -> np.ndarray:
    m = _blank()
    cv2.rectangle(m, (200 - a // 2, 200 - a // 2), (200 + a // 2, 200 + a // 2), 255, -1)
    return m


def rect_mask(w: int = 300, h: int = 150) -> np.ndarray:
    m = _blank()
    cv2.rectangle(m, (200 - w // 2, 200 - h // 2), (200 + w // 2, 200 + h // 2), 255, -1)
    return m


def polygon_mask(n: int, r: int = 150) -> np.ndarray:
    m = _blank()
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([200 + r * np.cos(ang), 200 + r * np.sin(ang)], axis=1)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m


def k_of(mask: np.ndarray) -> tuple[float, float]:
    cnt = _contour(mask)
    k = formal_k(cnt, SHAPE)
    return k, effective_k(k, rect_fill_score(cnt))


def test_k_values() -> None:
    cases = {
        "круг": (circle_mask(), 0.97, 1.01),
        "квадрат": (square_mask(), 0.68, 0.74),
        "прямоугольник 2:1": (rect_mask(), 0.40, 0.50),
        "шестиугольник": (polygon_mask(6), 0.83, 0.90),
        "восьмиугольник": (polygon_mask(8), 0.89, 0.95),
    }
    for name, (mask, lo, hi) in cases.items():
        k, keff = k_of(mask)
        assert lo <= k <= hi, f"{name}: K={k:.3f} вне [{lo}, {hi}]"
        want_round = name not in ("квадрат", "прямоугольник 2:1")
        got_round = keff >= CIRCLE_RATIO_THRESHOLD
        assert got_round == want_round, f"{name}: K_eff={keff:.3f} round={got_round}"


def test_size_rules() -> None:
    # ящик 400×400×300 из вопроса №1 — сторона 400 > 320 → C
    assert classify((400.0, 400.0), 0.5, 300.0) is Category.SIZE
    # тот же ящик в другой ориентации — тот же результат
    assert classify((400.0, 300.0), 0.5, 400.0) is Category.SIZE
    # проходит по всем трём: 440×310×300
    assert classify((440.0, 310.0), 0.5, 300.0) is Category.OK
    # любая сторона < 10 мм → C (вопросы 54, 108, 111)
    assert classify((150.0, 8.0), 0.5, 8.0) is Category.SIZE
    assert classify((150.0, 30.0), 0.5, 9.0) is Category.SIZE
    # без измеренной высоты проверяем только две стороны
    assert classify((300.0, 200.0), 0.5) is Category.OK
    assert classify((500.0, 200.0), 0.5) is Category.SIZE


def test_circle_rule() -> None:
    # мешок из вопроса №113: K = 0.817 → D
    assert classify((200.0, 180.0), 0.817, 150.0) is Category.CIRCLE
    # чуть ниже порога → B
    assert classify((200.0, 180.0), 0.79, 150.0) is Category.OK
    # круг в сечении не спасает от негабарита: габариты проверяются первыми
    assert classify((600.0, 180.0), 0.95, 150.0) is Category.SIZE
    # лежащий цилиндр: сверху прямоугольник (K низкий), сбоку круг → D
    assert classify((200.0, 60.0), max(0.45, 0.98), 60.0) is Category.CIRCLE


def test_marker_is_not_size_reject() -> None:
    """Тонкий предмет 150×20×20 больше не уходит в C по бывшему stick_small."""
    assert classify((150.0, 20.0), 0.5, 20.0) is Category.OK
    # но если он цилиндр — боковая проекция даёт круг → D
    assert classify((150.0, 20.0), 0.99, 20.0) is Category.CIRCLE


def _flat_det(long_mm: float, short_mm: float, k_xy: float) -> Detection:
    cnt = np.array(
        [[[300, 300]], [[500, 300]], [[500, 360]], [[300, 360]]], dtype=np.int32
    )
    return Detection(
        contour=cnt,
        bbox=(300, 300, 200, 60),
        size_mm=(long_mm, short_mm),
        circle_ratio=k_xy,
        category=Category.OK,
        area_px=12000.0,
        rect_fill=0.95,
        r_in_out=k_xy,
        k_xy=k_xy,
    )


def _run_passage(det: Detection, side: tuple[float, float] | None) -> Detection | None:
    dec = PassageDecider()
    frame = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(5):
        out = dec.feed(
            frame, det, track_miss=0, roi_xy=(320.0, 200.0), roi_hw=(400, 640)
        )
        assert out is None
        if side is not None:
            dec.feed_side(side[0], side[1])
    for _ in range(dec.end_idle):
        out = dec.feed(frame, None)
        if out is not None:
            return out
    return None


def test_lying_cylinder_needs_side_view() -> None:
    """Сверху цилиндр на боку — прямоугольник (B), сбоку круг → D."""
    det = _flat_det(200.0, 60.0, 0.45)
    only_top = _run_passage(det, None)
    assert only_top is not None and only_top.category is Category.OK
    with_side = _run_passage(det, (0.96, 60.0))
    assert with_side is not None
    assert with_side.category is Category.CIRCLE, with_side.category
    assert with_side.height_mm == 60.0
    assert with_side.k_side == 0.96


def test_side_view_catches_tall_object() -> None:
    """Ящик 400×300 сверху проходит, но с высотой 400 мм — уже C (вопрос №1)."""
    det = _flat_det(400.0, 300.0, 0.45)
    assert _run_passage(det, None).category is Category.OK
    tall = _run_passage(det, (0.4, 400.0))
    assert tall is not None and tall.category is Category.SIZE, tall.category
    # 300 мм высоты те же габариты уже проходят
    fits = _run_passage(det, (0.4, 300.0))
    assert fits is not None and fits.category is Category.OK, fits.category


if __name__ == "__main__":
    for fn in (
        test_k_values,
        test_size_rules,
        test_circle_rule,
        test_marker_is_not_size_reject,
        test_lying_cylinder_needs_side_view,
        test_side_view_catches_tall_object,
    ):
        fn()
        print(f"ok  {fn.__name__}")
    for name, mask in (
        ("круг", circle_mask()),
        ("квадрат", square_mask()),
        ("прямоуг 2:1", rect_mask()),
        ("6-угольник", polygon_mask(6)),
        ("8-угольник", polygon_mask(8)),
    ):
        k, keff = k_of(mask)
        cnt = _contour(mask)
        print(
            f"{name:>12}: K={k:.3f} K_eff={keff:.3f} "
            f"rect_fill={rect_fill_score(cnt):.3f}"
        )
