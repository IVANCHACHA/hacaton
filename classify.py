#!/usr/bin/env python3
"""
Классификация товаров (OpenCV) + CLI-режим.

Категории:
  B — подходит для сортировки
  C — не подходит по габаритам
  D — не подходит без доупаковки (круг в сечении)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from camera import find_iphone_source, list_avfoundation_devices, open_capture
from segment_refine import ContourRefiner, get_refiner

# Постановка задачи (трек 3): B — товар больше 10×10×10 мм и меньше 450×320×320 мм,
# при этом ни в одной 2D-проекции (XY, XZ, YZ) нет «круга в сечении».
# Круг определяется как K = r_вписанной / R_описанной ≥ 0.8 (ответ экспертов №73, №113).
MIN_SIZE_MM = 10.0
MAX_DIMS_MM = (450.0, 320.0, 320.0)
CIRCLE_RATIO_THRESHOLD = 0.80
# площадь/minAreaRect: у круга ≈0.785, поэтому ≥0.90 — заведомо многоугольник
RECT_FILL_NOT_ROUND = 0.90
ROI_SCALE = 0.72

# Нормализованный ROI (TL, TR, BR, BL), координаты 0..1 — зона ленты
DEFAULT_ROI_QUAD = (
    (0.301, 0.0),
    (0.748, 0.0),
    (0.721, 0.554),
    (0.285, 0.547),
)


class Category(Enum):
    OK = "Подходит для сортировки"
    SIZE = "Не подходит для сортировки по габаритам"
    CIRCLE = "Не подходит для сортировки без доупаковки"


ZONE = {
    Category.OK: "B",
    Category.SIZE: "C",
    Category.CIRCLE: "D",
}

COLOR = {
    Category.OK: (60, 180, 75),
    Category.SIZE: (0, 165, 255),
    Category.CIRCLE: (0, 0, 255),
}


@dataclass
class Detection:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    size_mm: tuple[float, float]
    circle_ratio: float
    category: Category
    area_px: float
    # вписанная / описанная окружность в координатах полного кадра
    circle_in: tuple[tuple[float, float], float] | None = None
    circle_out: tuple[tuple[float, float], float] | None = None
    rect_fill: float = 0.0  # area/minAreaRect — ~1 у коробки, ~0.78 у круга
    r_in_out: float = 0.0  # сырой r_in/r_out без ellipse-boost
    k_xy: float = 0.0  # K = r_вп/R_оп по виду сверху (критерий постановки)
    k_side: float | None = None  # то же по боковой проекции
    height_mm: float | None = None
    track_id: int | None = None

    def to_dict(self) -> dict:
        a, b = self.size_mm
        x, y, w, h = self.bbox
        data = {
            "zone": ZONE[self.category],
            "category": self.category.value,
            "size_mm": [round(a, 1), round(b, 1)],
            "circle_ratio": round(self.circle_ratio, 3),
            "bbox": [x, y, w, h],
            "area_px": round(self.area_px, 1),
            "rect_fill": round(self.rect_fill, 3),
            "r_in_out": round(self.r_in_out, 3),
            "k_xy": round(self.k_xy, 3),
        }
        if self.k_side is not None:
            data["k_side"] = round(self.k_side, 3)
        if self.height_mm is not None:
            data["height_mm"] = round(self.height_mm, 1)
        if self.track_id is not None:
            data["track_id"] = self.track_id
        if self.circle_in:
            (cx, cy), r = self.circle_in
            data["circle_in"] = {"cx": round(cx, 1), "cy": round(cy, 1), "r": round(r, 1)}
        if self.circle_out:
            (cx, cy), r = self.circle_out
            data["circle_out"] = {"cx": round(cx, 1), "cy": round(cy, 1), "r": round(r, 1)}
        return data


def roi_rect(h: int, w: int) -> tuple[int, int, int, int]:
    rw, rh = int(w * ROI_SCALE), int(h * ROI_SCALE)
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    return x0, y0, rw, rh


def normalize_quad(quad) -> list[tuple[float, float]]:
    if not quad or len(quad) != 4:
        return [tuple(p) for p in DEFAULT_ROI_QUAD]
    out: list[tuple[float, float]] = []
    for p in quad:
        x = float(p[0])
        y = float(p[1])
        out.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return out


def quad_to_pixels(h: int, w: int, quad) -> np.ndarray:
    pts = normalize_quad(quad)
    return np.array([[x * (w - 1), y * (h - 1)] for x, y in pts], dtype=np.float32)


def warp_roi(frame: np.ndarray, quad_px: np.ndarray):
    """Перспективно выпрямляет четырёхугольник → прямоугольник."""
    pts = quad_px.astype(np.float32)
    w1 = float(np.linalg.norm(pts[1] - pts[0]))
    w2 = float(np.linalg.norm(pts[2] - pts[3]))
    h1 = float(np.linalg.norm(pts[3] - pts[0]))
    h2 = float(np.linalg.norm(pts[2] - pts[1]))
    rw = max(32, int(max(w1, w2)))
    rh = max(32, int(max(h1, h2)))
    dst = np.array(
        [[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    inv = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(frame, matrix, (rw, rh))
    return warped, inv, rw, rh


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Модуль градиента Sobel (float32)."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _seam_rows(mag: np.ndarray, thr: float) -> np.ndarray:
    """Строки ROI, где градиент тянется почти через всю ширину — швы ленты."""
    rh, rw = mag.shape[:2]
    frac = (mag > thr).mean(axis=1)
    seam = frac > 0.45
    # шов тонкий: расширяем на пару пикселей вокруг
    seam = cv2.dilate(
        seam.astype(np.uint8).reshape(-1, 1), np.ones((5, 1), np.uint8), iterations=1
    ).ravel().astype(bool)
    if int(seam.sum()) > 0.5 * rh:
        return np.zeros(rh, dtype=bool)
    return seam


def _sobel_edges(gray: np.ndarray, belt: np.ndarray) -> np.ndarray:
    """Границы Sobel без швов ленты (для замыкания силуэта товара)."""
    mag = _sobel_magnitude(gray)
    ref_pool = mag[belt] if int(belt.sum()) >= 200 else mag
    ref = float(np.median(ref_pool))
    p95 = float(np.percentile(ref_pool, 95))
    thr = max(26.0, ref * 2.6 + 12.0, p95 * 1.05)
    edge = (mag > thr).astype(np.uint8) * 255
    # швы ленты по всей ширине — не граница товара
    edge[_seam_rows(mag, thr), :] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, k, iterations=1)
    edge = cv2.subtract(edge, _thin_gap_mask(edge, horizontal=True, max_thick=6))
    return edge


def _belt_background(roi_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Фон ленты по строкам (гасит слябы/швы и градиент света) + глобальная медиана."""
    blur = cv2.GaussianBlur(roi_bgr, (3, 3), 0).astype(np.float32)
    glob = np.median(blur.reshape(-1, 3), axis=0)
    row_med = np.median(blur, axis=1)
    rh = row_med.shape[0]
    ys = np.arange(rh, dtype=np.float32)

    # плавный тренд освещённости вдоль ленты (робастно, чтобы крупный товар не тянул)
    trend = np.empty_like(row_med)
    for c in range(3):
        vals = row_med[:, c]
        wts = np.ones(rh, dtype=np.float32)
        coef = np.polyfit(ys, vals, 2, w=wts)
        for _ in range(2):
            fit = np.polyval(coef, ys)
            res = np.abs(vals - fit)
            scale = float(np.median(res)) + 1.0
            wts = (res < 3.0 * scale).astype(np.float32) + 1e-3
            coef = np.polyfit(ys, vals, 2, w=wts)
        trend[:, c] = np.polyval(coef, ys)

    # строку занял крупный товар → фон берём из тренда
    dev = np.linalg.norm(row_med - trend, axis=1)
    row_bg = row_med.copy()
    row_bg[dev > 26.0] = trend[dev > 26.0]
    row_bg = cv2.GaussianBlur(row_bg.reshape(-1, 1, 3), (1, 5), 0).reshape(-1, 3)
    bg = np.repeat(row_bg[:, None, :], blur.shape[1], axis=1)
    return bg, glob


def _appearance_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """Силуэт товара на текстурной ленте: отличие от построчного фона + Sobel."""
    blur = cv2.GaussianBlur(roi_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (h >= 35) & (h <= 95) & (s >= 45) & (v >= 40)

    rh, rw = v.shape
    my, mx = max(6, rh // 18), max(6, rw // 18)
    core = np.zeros_like(v, dtype=bool)
    core[my : rh - my, mx : rw - mx] = True
    belt = core & ~green
    if int(belt.sum()) < 200:
        belt = core

    bg, _glob = _belt_background(roi_bgr)
    bg_v = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    bg_s = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)

    vf = v.astype(np.float32)
    sf = s.astype(np.float32)
    # тень = тот же оттенок, просто темнее → сравниваем хроматику, а не яркость
    chroma = _chroma_distance(blur.astype(np.float32), bg)
    ratio = vf / np.maximum(bg_v, 1.0)
    by_bright = vf > (bg_v + 22.0)
    by_dark = ratio < 0.68
    by_chroma = chroma > 13.0
    by_sat = (sf > bg_s + 28.0) & (vf > 40.0)

    th = (by_bright | by_dark | by_chroma | by_sat).astype(np.uint8) * 255
    # Sobel замыкает силуэт только рядом с цветовой опорой (не сам по себе)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sobel = _sobel_edges(gray, belt)
    near = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    th = cv2.bitwise_or(th, cv2.bitwise_and(sobel, near))

    # боковые направляющие ленты часто зелёные — режем края ROI
    edge = max(3, min(rh, rw) // 40)
    th[:, :edge] = 0
    th[:, rw - edge :] = 0

    # мягкий open — не съедать карандаш (~8–15 px толщины)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    # склеиваем части, но не раздуваем в ленту
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k2, iterations=1)
    # ещё раз режем тонкие щели после склейки
    th = cv2.subtract(th, _thin_gap_mask(th, horizontal=True, max_thick=8))
    th = cv2.subtract(th, _thin_gap_mask(th, horizontal=False, max_thick=8))
    filled = _fill_holes(th)
    if float(np.mean(filled)) > 0.40 * 255:
        return th
    return filled


def _chroma_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Разница цвета без влияния яркости — тень даёт ≈0."""
    pa = a / np.maximum(a.sum(axis=2, keepdims=True), 1.0)
    pb = b / np.maximum(b.sum(axis=2, keepdims=True), 1.0)
    return np.linalg.norm(pa - pb, axis=2) * 255.0


def _belt_distance_map(roi_bgr: np.ndarray) -> np.ndarray:
    """Отличие пикселя от фона ленты: хроматика + «пересветы/провалы» яркости."""
    blur = cv2.GaussianBlur(roi_bgr, (3, 3), 0).astype(np.float32)
    bg, _glob = _belt_background(roi_bgr)
    chroma = _chroma_distance(blur, bg)
    lum = cv2.cvtColor(blur.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_lum = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    # тень (просто темнее) в счёт не идёт; ярче фона или радикально темнее — идёт
    up = np.maximum(lum - bg_lum, 0.0)
    down = np.maximum(bg_lum * 0.68 - lum, 0.0)
    return chroma * 2.0 + up + down


def _blob_contrast(roi_bgr: np.ndarray, cnt: np.ndarray, dist_map: np.ndarray) -> tuple[float, float]:
    """Насколько внутренность блоба отличается от ленты: (среднее, доля > 22)."""
    rh, rw = roi_bgr.shape[:2]
    mask = np.zeros((rh, rw), np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    if int((inner > 0).sum()) < 30:
        inner = mask
    pool = dist_map[inner > 0]
    if pool.size == 0:
        return 0.0, 0.0
    return float(pool.mean()), float((pool > 22.0).mean())


def _trim_appendages(cnt: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Убирает тонкие хвосты/мостики маски, оставляя тело объекта."""
    rh, rw = shape_hw
    mask = np.zeros((rh, rw), np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    r_max = float(dt.max())
    if r_max < 4.0:
        return cnt
    core = (dt > 0.45 * r_max).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    if n <= 1:
        return cnt
    lab = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]))
    grown = (labels == lab).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(24):
        nxt = cv2.bitwise_and(cv2.dilate(grown, k, iterations=1), mask)
        if int(cv2.countNonZero(cv2.absdiff(nxt, grown))) == 0:
            break
        grown = nxt
    base_area = float(cv2.contourArea(cnt))
    new_area = float(cv2.countNonZero(grown))
    if new_area < 0.5 * base_area or new_area > 1.02 * base_area:
        return cnt
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cnt
    return max(cnts, key=cv2.contourArea)


def _snap_contour(roi_bgr: np.ndarray, cnt: np.ndarray, dist_map: np.ndarray) -> np.ndarray:
    """Уточнение границы в окне объекта по отличию от ленты (тень не считается)."""
    rh, rw = roi_bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad = int(max(8, 0.18 * max(bw, bh)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(rw, x + bw + pad), min(rh, y + bh + pad)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return cnt
    crop = cv2.GaussianBlur(roi_bgr[y0:y1, x0:x1], (3, 3), 0)
    ch, cw = crop.shape[:2]
    dist = dist_map[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grad = _sobel_magnitude(gray)
    feat = dist + 0.25 * np.clip(grad, 0, 120)
    fmax = float(np.percentile(feat, 99.5))
    f8 = np.clip(feat / max(fmax, 1e-3) * 255.0, 0, 255).astype(np.uint8)
    thr, mask = cv2.threshold(f8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # порог не ниже абсолютного минимума по контрасту к ленте
    mask[feat < 18.0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = _fill_holes(mask)

    m = cv2.moments(cnt)
    if m["m00"] < 1e-6:
        return cnt
    cx = int(np.clip(round(m["m10"] / m["m00"]) - x0, 0, cw - 1))
    cy = int(np.clip(round(m["m01"] / m["m00"]) - y0, 0, ch - 1))
    base_area = float(cv2.contourArea(cnt))
    comp = _component_at(mask, cx, cy, max(60.0, 0.2 * base_area))
    if comp is None:
        return cnt
    new_area = float(cv2.contourArea(comp))
    if new_area < 0.35 * base_area or new_area > 1.7 * base_area:
        return cnt
    nx, ny, nbw, nbh = cv2.boundingRect(comp)
    # не принимаем маску, вросшую в границы окна (значит утекла в ленту)
    if nx <= 1 or ny <= 1 or nx + nbw >= cw - 1 or ny + nbh >= ch - 1:
        return cnt
    return comp + np.array([[[x0, y0]]], np.int32)


def _order_rect_points(box: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL для вытянутого прямоугольника (длинная сторона горизонтально)."""
    pts = np.asarray(box, dtype=np.float32)
    # два ребра от точки с мин. y+x ≈ TL
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    # если вертикальная ориентация — повернуть на 90°
    w = float(np.linalg.norm(ordered[1] - ordered[0]))
    h = float(np.linalg.norm(ordered[3] - ordered[0]))
    if h > w:
        ordered = np.array([ordered[3], ordered[0], ordered[1], ordered[2]], dtype=np.float32)
    return ordered


def _scale_length_from_ticks(strip_bgr: np.ndarray) -> tuple[float, int] | None:
    """Длина шкалы 0→конец по рискам (не корпус линейки)."""
    th, tw = strip_bgr.shape[:2]
    if tw < 80 or th < 8:
        return None
    band_h = max(10, int(0.40 * th))
    band = strip_bgr[:band_h]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    thr_v = float(np.median(gray) - 18.0)
    bw = (gray.astype(np.float32) < thr_v).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(4, band_h // 4)))
    ticks = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)
    proj = ticks.sum(axis=0).astype(np.float64)
    proj = cv2.GaussianBlur(proj.reshape(1, -1), (0, 0), 0.7).ravel()
    if float(proj.max()) < 1.0:
        return None
    proj /= float(proj.max())
    margin = max(2, int(0.01 * tw))
    peaks: list[int] = []
    for i in range(margin + 1, tw - margin - 1):
        if proj[i] < 0.25:
            continue
        if proj[i] >= proj[i - 1] and proj[i] >= proj[i + 1]:
            if not peaks or i - peaks[-1] >= 3:
                peaks.append(i)
            elif proj[i] > proj[peaks[-1]]:
                peaks[-1] = i
    if len(peaks) < 8:
        return None

    # подбираем пару пиков = 0 и последнее деление (20 см → 21 риск)
    expect = tw / 20.0
    best: tuple[float, float, float, int] | None = None
    for i, a in enumerate(peaks):
        for b in peaks[i + 5 :]:
            span = float(b - a)
            if span < 0.78 * tw or span > 0.995 * tw:
                continue
            pitch = span / 20.0
            if not (0.55 * expect <= pitch <= 1.45 * expect):
                continue
            hits = 0
            for k in range(21):
                x = a + k * pitch
                if any(abs(p - x) <= max(2.0, 0.35 * pitch) for p in peaks):
                    hits += 1
            if hits < 12:
                continue
            score = (float(hits), span)
            if best is None or score > (best[3], best[0]):
                best = (span, float(a), float(b), hits)
    if best is None:
        return None
    span, _a, _b, hits = best
    return span, int(hits)


def calibrate_mm_per_px(
    frame: np.ndarray,
    roi_quad=None,
    ruler_mm: float = 200.0,
) -> dict | None:
    """Масштаб по зелёной линейке: ruler_mm = длина шкалы (0→последнее деление), не корпус."""
    h, w = frame.shape[:2]
    quad_px = quad_to_pixels(h, w, roi_quad)
    roi, _, _, _ = warp_roi(frame, quad_px)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 45, 45), (95, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 2000:
        return None
    rect = cv2.minAreaRect(cnt)
    (_cx, _cy), (ww, hh), _ang = rect
    body_px = float(max(ww, hh))
    minor = float(min(ww, hh))
    if body_px < 80 or minor < 12 or body_px / max(minor, 1.0) < 3.0:
        return None

    box = cv2.boxPoints(rect)
    ordered = _order_rect_points(box)
    length = int(round(np.linalg.norm(ordered[1] - ordered[0])))
    thick = int(round(np.linalg.norm(ordered[3] - ordered[0])))
    length = max(80, length)
    thick = max(12, thick)
    dst = np.array(
        [[0, 0], [length - 1, 0], [length - 1, thick - 1], [0, thick - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(ordered, dst)
    strip = cv2.warpPerspective(roi, M, (length, thick))

    tick = _scale_length_from_ticks(strip)
    if tick is not None:
        scale_px, n_ticks = tick
        method = "scale"
    else:
        # корпус чуть длиннее шкалы (~21 см при шкале 20 см)
        scale_px = body_px * (float(ruler_mm) / (float(ruler_mm) + 14.0))
        n_ticks = 0
        method = "body_inset"

    return {
        "mm_per_px": float(ruler_mm) / float(scale_px),
        "scale_px": float(scale_px),
        "body_px": float(body_px),
        "ticks": int(n_ticks),
        "method": method,
        "ruler_mm": float(ruler_mm),
    }


def _edge_touches(x: int, y: int, bw: int, bh: int, rw: int, rh: int, margin: int = 3) -> int:
    n = 0
    if x <= margin:
        n += 1
    if y <= margin:
        n += 1
    if x + bw >= rw - margin:
        n += 1
    if y + bh >= rh - margin:
        n += 1
    return n


def _is_belt_gap_px(
    min_side: float,
    max_side: float,
    *,
    area: float,
    bw: int,
    bh: int,
    rw: int,
    rh: int,
    touches: int,
) -> bool:
    """Горизонтальная/вертикальная щель модульной ленты."""
    aspect = max_side / max(1e-3, min_side)
    extent = area / max(1.0, float(bw * bh))
    # полоса почти через ROI
    if bw >= int(0.38 * rw) and bh < 32:
        return True
    if bh >= int(0.38 * rh) and bw < 32:
        return True
    if min_side < 20.0 and max_side > 0.48 * max(rw, rh):
        return True
    if touches >= 2 and min_side < 30.0:
        return True
    # рваная длинная маска — типичная щель
    if aspect >= 3.0 and extent < 0.38 and max_side > 70.0:
        return True
    fill_len = area / max(max_side, 1.0)
    if min_side < 22.0 and fill_len < 0.50 * min_side and max_side > 70.0:
        return True
    return False


def _is_stick_px(
    min_side: float,
    max_side: float,
    *,
    area: float,
    rw: int,
    rh: int,
    touches: int,
) -> bool:
    """Карандаш: плотный стержень, не щель через ленту."""
    if min_side < 14.0 or max_side < 130.0:
        return False
    aspect = max_side / max(1e-3, min_side)
    if not (5.0 <= aspect <= 14.0):
        return False
    if touches >= 2:
        return False
    if max_side > 0.55 * max(rw, rh):
        return False
    extent = area / max(1.0, min_side * max_side)
    if extent < 0.42:
        return False
    fill_len = area / max(max_side, 1.0)
    if fill_len < 0.65 * min_side:
        return False
    return True


def _thin_gap_mask(fg: np.ndarray, *, horizontal: bool, max_thick: int = 8) -> np.ndarray:
    """Полосы-щели (тоньше карандаша)."""
    if horizontal:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        thick_idx = cv2.CC_STAT_HEIGHT
    else:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        thick_idx = cv2.CC_STAT_WIDTH
    cand = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    out = np.zeros_like(cand)
    for i in range(1, n):
        if int(stats[i, thick_idx]) <= max_thick:
            out[labels == i] = 255
    return out


def _component_at(mask: np.ndarray, cx: float, cy: float, min_area: float) -> np.ndarray | None:
    """Контур той компоненты appearance, которая содержит точку трека."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    x = int(np.clip(round(cx), 0, mask.shape[1] - 1))
    y = int(np.clip(round(cy), 0, mask.shape[0] - 1))
    lab = int(labels[y, x])
    if lab == 0 or float(stats[lab, cv2.CC_STAT_AREA]) < min_area:
        best_i, best_d = 0, 1e18
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            bx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] * 0.5
            by = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] * 0.5
            d = (bx - cx) ** 2 + (by - cy) ** 2
            if d < best_d:
                best_d, best_i = d, i
        lab = best_i
    if lab == 0:
        return None
    blob = np.zeros_like(mask)
    blob[labels == lab] = 255
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _assemble_object(mask: np.ndarray, seed_cnt: np.ndarray, min_area: float) -> np.ndarray:
    """Склеивает соседние части одного товара без раздувания в ленту."""
    rh, rw = mask.shape[:2]
    seed = np.zeros((rh, rw), np.uint8)
    cv2.drawContours(seed, [seed_cnt], -1, 255, -1)
    rad = int(max(18, 0.11 * min(rh, rw)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad | 1, rad | 1))
    gate = cv2.dilate(seed, k, iterations=1)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    sm = cv2.moments(seed_cnt)
    if sm["m00"] < 1e-6:
        return seed_cnt
    sx = float(sm["m10"] / sm["m00"])
    sy = float(sm["m01"] / sm["m00"])
    seed_area = float(cv2.contourArea(seed_cnt))

    merged = seed.copy()
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area * 0.4:
            continue
        part = labels == i
        if not np.any(gate[part]):
            cx, cy = float(cents[i][0]), float(cents[i][1])
            if (cx - sx) ** 2 + (cy - sy) ** 2 > (rad * 1.15) ** 2:
                continue
        if area > 1.8 * max(seed_area, min_area):
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if _edge_touches(bx, by, bw, bh, rw, rh, margin=2) >= 2:
            continue
        merged[part] = 255

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k2, iterations=1)
    merged = cv2.bitwise_and(merged, mask)
    merged = _fill_holes(merged)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return seed_cnt
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    # hull только если почти не растёт (иначе захватывает фон между рукоятками)
    if hull_area > 1 and 0.45 <= area / hull_area < 0.72 and hull_area <= area * 1.35:
        return hull
    return cnt


def _ellipse_contour(ellipse, n: int = 72) -> np.ndarray:
    (cx, cy), (ma, mb), angle = ellipse
    pts = cv2.ellipse2Poly(
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(ma * 0.5))), max(1, int(round(mb * 0.5)))),
        int(round(angle)),
        0,
        360,
        max(1, 360 // n),
    )
    return pts.reshape(-1, 1, 2).astype(np.int32)


def _smooth_contour(cnt: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Сглаживает край без раздувания площади."""
    h, w = shape_hw
    if cnt is None or len(cnt) < 3:
        return cnt
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return cnt
    smooth = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(smooth))
    (_x, _y), r = cv2.minEnclosingCircle(smooth)
    if r > 1 and area / (np.pi * r * r) >= 0.70 and len(smooth) >= 5:
        return _ellipse_contour(cv2.fitEllipse(smooth.astype(np.float32)))
    peri = cv2.arcLength(smooth, True)
    approx = cv2.approxPolyDP(smooth, max(1.0, 0.006 * peri), True)
    return approx if len(approx) >= 3 else smooth


def inscribed_circle(
    contour: np.ndarray, shape_hw: tuple[int, int]
) -> tuple[tuple[float, float], float]:
    """Центр и радиус наибольшей вписанной окружности (distance transform)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # шайба/кольцо: дырка не должна ронять вписанный радиус
    mask = _fill_holes(mask)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val < 1:
        m = cv2.moments(contour)
        if m["m00"] > 1e-6:
            return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])), 1.0
        return (0.0, 0.0), 0.0
    return (float(max_loc[0]), float(max_loc[1])), float(max_val)


def circumscribed_circle(contour: np.ndarray) -> tuple[tuple[float, float], float]:
    (x, y), r = cv2.minEnclosingCircle(contour)
    return (float(x), float(y)), float(r)


def contour_roundness(contour: np.ndarray) -> float:
    """area / (π R²) по minEnclosingCircle — близко к 1 для диска/шайбы."""
    area = float(cv2.contourArea(contour))
    if area < 1:
        return 0.0
    (_x, _y), r = cv2.minEnclosingCircle(contour)
    if r < 1:
        return 0.0
    return float(area / (np.pi * r * r))


def contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if area < 1 or peri < 1:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def ellipse_aspect(contour: np.ndarray) -> float:
    if contour is None or len(contour) < 5:
        x, y, bw, bh = cv2.boundingRect(contour)
        return float(max(bw, bh) / max(1, min(bw, bh)))
    (_c, (ma, mb), _a) = cv2.minAreaRect(contour.astype(np.float32))
    return float(max(ma, mb) / max(1e-3, min(ma, mb)))


def ellipse_fill(contour: np.ndarray) -> tuple[float, float]:
    """Насколько контур = эллипс (fill, aspect). Учитывает перспективу ROI."""
    if contour is None or len(contour) < 5:
        return 0.0, 99.0
    area = float(cv2.contourArea(contour))
    if area < 1:
        return 0.0, 99.0
    (_c, (ma, mb), _a) = cv2.fitEllipse(contour.astype(np.float32))
    short, long = float(min(ma, mb)), float(max(ma, mb))
    if short < 1:
        return 0.0, 99.0
    asp = long / short
    el_area = float(np.pi * (ma * 0.5) * (mb * 0.5))
    fill = area / max(1.0, el_area)
    # >1 из‑за зубцов — клип
    return float(min(fill, 1.2)), float(asp)


def _solid_contour(contour: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Внешний контур с заливкой дырок (шайба → диск)."""
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = _fill_holes(mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return contour
    return max(cnts, key=cv2.contourArea)


def rect_fill_score(contour: np.ndarray) -> float:
    """area / minAreaRect — близко к 1 у колоды/коробки, ~0.78 у круга."""
    area = float(cv2.contourArea(contour))
    if area < 1 or len(contour) < 3:
        return 0.0
    (_c, (ma, mb), _a) = cv2.minAreaRect(contour.astype(np.float32))
    ra = float(max(ma, 1.0) * max(mb, 1.0))
    return float(min(1.2, area / ra))


def circle_likeness(contour: np.ndarray, shape_hw: tuple[int, int]) -> tuple[float, tuple, float, tuple, float, float, float]:
    """Круг/шайба → высокий score. Возвращает также rect_fill и сырой r_in/r_out."""
    # boxy считаем до сильного close — иначе прямоугольник «округляется»
    rfill_raw = rect_fill_score(contour)
    peri0 = float(cv2.arcLength(contour, True)) if contour is not None and len(contour) >= 3 else 0.0
    approx0 = (
        cv2.approxPolyDP(contour, max(2.0, 0.035 * peri0), True)
        if peri0 > 0
        else contour
    )
    n_vert0 = len(approx0) if approx0 is not None else 99

    solid = _solid_contour(contour, shape_hw)
    c_out, r_out = circumscribed_circle(solid)
    c_in, r_in = inscribed_circle(solid, shape_hw)
    ratio = circle_ratio_from_radii(r_in, r_out)
    rnd = contour_roundness(solid)
    circ = contour_circularity(solid)
    el_fill, el_asp = ellipse_fill(solid)
    rfill = max(rfill_raw, rect_fill_score(solid))
    peri = float(cv2.arcLength(solid, True))
    approx = cv2.approxPolyDP(solid, max(2.0, 0.04 * peri), True)
    n_vert = len(approx)

    # прямоугольник/колода: плотно заполняет minAreaRect (круг ~0.78)
    boxy = rfill_raw >= 0.86 or rfill >= 0.88
    if n_vert0 <= 6 and rfill_raw >= 0.82:
        boxy = True
    if n_vert <= 6 and rfill >= 0.84:
        boxy = True
    # вытянутый «кирпич»: низкий r_in/r_out при высоком rect_fill
    if rfill_raw >= 0.83 and ratio < 0.62 and rnd < 0.68:
        boxy = True

    score = max(ratio, rnd)
    if ratio >= 0.72 and rnd >= 0.70:
        score = max(score, 0.5 * (ratio + rnd))
    if ratio >= 0.78 and rnd >= 0.78 and circ >= 0.72:
        score = max(score, 0.92)

    # эллипс после warp — только если НЕ коробка и rect_fill как у эллипса
    if not boxy and el_fill >= 0.82 and el_asp <= 2.45 and rfill <= 0.84:
        ell = 0.70 + 0.30 * min(1.0, (el_fill - 0.82) / 0.16)
        if el_asp > 1.9:
            ell *= max(0.82, 1.15 - 0.15 * (el_asp - 1.9))
        # требуем хоть какую-то круглость / гладкость
        if circ >= 0.60 or ratio >= 0.55:
            score = max(score, ell)
    if not boxy and el_fill >= 0.88 and el_asp <= 2.15 and rfill <= 0.83 and (ratio >= 0.52 or circ >= 0.65):
        score = max(score, 0.86)
    if not boxy and el_fill >= 0.92 and el_asp <= 1.85 and rfill <= 0.82 and ratio >= 0.60:
        score = max(score, 0.92)

    if boxy:
        score = min(score, 0.52)
    elif rfill >= 0.85 and ratio < 0.75:
        score = min(score, 0.58)

    return (
        float(min(1.0, max(0.0, score))),
        c_in,
        r_in,
        c_out,
        r_out,
        float(rfill_raw),
        float(ratio),
    )


def transform_circle(
    center: tuple[float, float],
    radius: float,
    inv: np.ndarray,
) -> tuple[tuple[float, float], float]:
    """Перенос окружности из ROI в полный кадр (радиус — по точке на окружности)."""
    c = np.array([[[center[0], center[1]]]], dtype=np.float32)
    p = np.array([[[center[0] + radius, center[1]]]], dtype=np.float32)
    c2 = cv2.perspectiveTransform(c, inv)[0, 0]
    p2 = cv2.perspectiveTransform(p, inv)[0, 0]
    r2 = float(np.linalg.norm(p2 - c2))
    return (float(c2[0]), float(c2[1])), max(0.0, r2)


def circle_ratio_from_radii(r_in: float, r_out: float) -> float:
    if r_out < 1e-6:
        return 0.0
    return float(r_in / r_out)


def formal_k(contour: np.ndarray, shape_hw: tuple[int, int]) -> float:
    """K = r_вписанной / R_описанной по 2D-проекции — критерий из постановки."""
    if contour is None or len(contour) < 3:
        return 0.0
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # отверстие (шайба) не должно уменьшать вписанную окружность
    mask = _fill_holes(mask)
    # снимаем только пиксельную «зубчатость» маски, форму не округляем
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    solid = max(cnts, key=cv2.contourArea)
    _c_out, r_out = circumscribed_circle(solid)
    if r_out < 1.0:
        return 0.0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    r_in = float(dist.max())
    return float(min(1.0, r_in / r_out))


def effective_k(k: float, rect_fill: float) -> float:
    """K с поправкой: плотно заполненный прямоугольник кругом быть не может."""
    if rect_fill >= RECT_FILL_NOT_ROUND:
        return min(k, CIRCLE_RATIO_THRESHOLD - 0.05)
    return k


def classify(
    size_mm: tuple[float, float],
    ratio: float,
    height_mm: float | None = None,
) -> Category:
    """B/C/D по правилам постановки. ratio — итоговый K по доступным проекциям."""
    dims = [float(size_mm[0]), float(size_mm[1])]
    if height_mm is not None and height_mm > 0:
        dims.append(float(height_mm))
    dims.sort(reverse=True)
    too_small = min(dims) < MIN_SIZE_MM
    limits = sorted(MAX_DIMS_MM, reverse=True)[: len(dims)]
    too_big = any(d > lim for d, lim in zip(dims, limits))
    if too_small or too_big:
        return Category.SIZE
    if ratio >= CIRCLE_RATIO_THRESHOLD:
        return Category.CIRCLE
    return Category.OK


def roi_centrality(cx: float, cy: float, rw: float, rh: float) -> float:
    """1.0 в центре ROI, →0 к краям."""
    if rw < 1 or rh < 1:
        return 0.0
    dx = (cx / rw) - 0.5
    dy = (cy / rh) - 0.5
    # узкое «окно» середины ленты
    dist = float(np.hypot(dx / 0.28, dy / 0.32))
    return float(np.exp(-0.5 * dist * dist))


def detection_quality(
    det: Detection,
    frame_hw: tuple[int, int],
    *,
    track_miss: int = 0,
    sam_score: float = 0.0,
    centrality: float = 0.5,
) -> float:
    """Оценка кадра: сильно предпочитаем объект в середине ROI."""
    a, b = det.size_mm
    short, long = float(min(a, b)), float(max(a, b))
    aspect = long / max(1e-3, short)
    is_stick = short >= 6.0 and long >= 100.0 and 5.0 <= aspect <= 14.0
    if short < 6.0:
        return -1.0
    # щель: рваный fill или тонкая длинная полоса
    if aspect >= 3.0 and float(det.rect_fill) < 0.55:
        return -1.0
    if short < 14.0 and long > 150.0 and det.circle_ratio < 0.55:
        return -1.0
    if not is_stick:
        if short < 12.0:
            return -1.0
        if short < 28.0 and aspect > 2.4 and det.circle_ratio < 0.70:
            return -1.0
        if short < 22.0 and aspect > 2.0:
            return -1.0
    h, w = frame_hw
    x, y, bw, bh = det.bbox
    margin = float(min(x, y, w - (x + bw), h - (y + bh)))
    edge = 1.0 if margin >= 14 else max(0.12, margin / 14.0)
    fill = det.area_px / max(1.0, float(w * h))
    fill_pen = 1.0 if fill < 0.28 else max(0.18, 1.4 - 2.2 * fill)
    asp = 1.0 if aspect < 3.5 else float(np.exp(-0.35 * (aspect - 3.5)))
    round_boost = 0.50 + 1.35 * float(det.circle_ratio)
    if is_stick:
        iso = 0.95
    else:
        iso = 1.0 if aspect <= 1.55 else (0.90 if aspect <= 2.2 else asp)
    held = 0.22 if track_miss > 0 else 1.0
    sam = 0.85 + 0.35 * min(1.0, max(0.0, sam_score))
    size_b = min(1.6, 0.35 + short / 45.0)
    # центр ROI — главный вес
    center = 0.35 + 1.9 * float(centrality) ** 1.4
    return float(
        det.area_px * edge * fill_pen * round_boost * iso * held * sam * size_b * center
    )


class PassageDecider:
    """Один ответ на объект: топ кадров из середины ROI → голосование."""

    def __init__(self, end_idle: int = 4, min_obs: int = 3, keep: int = 5) -> None:
        self.end_idle = end_idle
        self.min_obs = min_obs
        self.keep = keep
        self._active = False
        self._obs = 0
        self._mid_obs = 0
        self._idle = 0
        self._cands: list[tuple[float, Detection, np.ndarray]] = []
        self._fallback: tuple[float, Detection, np.ndarray] | None = None
        self._best: Detection | None = None
        self._final: Detection | None = None
        self._final_frame: np.ndarray | None = None
        self._side_k: list[float] = []
        self._side_h: list[float] = []

    def reset(self) -> None:
        self.__init__(end_idle=self.end_idle, min_obs=self.min_obs, keep=self.keep)

    def feed_side(self, k_side: float, height_mm: float) -> None:
        """Проба с боковой камеры для текущего прохода (XZ/YZ и высота)."""
        if not self._active:
            return
        if height_mm > 0:
            self._side_h.append(float(height_mm))
        self._side_k.append(float(k_side))

    def feed(
        self,
        frame: np.ndarray,
        det: Detection | None,
        *,
        track_miss: int = 0,
        sam_score: float = 0.0,
        roi_xy: tuple[float, float] | None = None,
        roi_hw: tuple[int, int] | None = None,
    ) -> Detection | None:
        """Обновить проход. Возвращает Detection один раз при финализации."""
        if det is None:
            self._idle += 1
            if self._active and self._idle >= self.end_idle:
                return self._finalize()
            return None

        if not self._active:
            self._begin()
        self._idle = 0

        centrality = 0.5
        if roi_xy is not None and roi_hw is not None:
            # roi_xy=(cx,cy) в ROI, roi_hw=(rh, rw)
            rh, rw = float(roi_hw[0]), float(roi_hw[1])
            centrality = roi_centrality(roi_xy[0], roi_xy[1], rw, rh)

        score = detection_quality(
            det,
            frame.shape[:2],
            track_miss=track_miss,
            sam_score=sam_score,
            centrality=centrality,
        )
        if score < 0:
            return None
        if track_miss == 0:
            self._obs += 1
            if centrality >= 0.50:
                self._mid_obs += 1

        item = (score, det, frame.copy())
        if self._fallback is None or score > self._fallback[0]:
            self._fallback = item

        # в топ — только кадры из середины ROI
        if centrality >= 0.48:
            self._cands.append(item)
            self._cands.sort(key=lambda t: t[0], reverse=True)
            self._cands = self._cands[: self.keep]
        self._best = (self._cands[0][1] if self._cands else self._fallback[1])
        return None

    def ui_dict(self) -> dict | None:
        if self._final is not None:
            d = self._final.to_dict()
            d["final"] = True
            d["pending"] = False
            return d
        if self._active and self._best is not None:
            d = self._best.to_dict()
            d["final"] = False
            d["pending"] = True
            return d
        return None

    def final_frame(self) -> np.ndarray | None:
        return self._final_frame

    def _begin(self) -> None:
        self._active = True
        self._obs = 0
        self._mid_obs = 0
        self._idle = 0
        self._cands = []
        self._fallback = None
        self._best = None
        self._final = None
        self._final_frame = None
        self._side_k = []
        self._side_h = []

    def _finalize(self) -> Detection | None:
        self._active = False
        need = self.min_obs
        if self._mid_obs >= 2:
            need = min(need, 2)
        # без кадров из середины — не финализируем (края/шум ленты)
        pool = list(self._cands)
        if not pool or self._mid_obs < 1 or self._obs < need:
            self._cands = []
            self._fallback = None
            self._best = None
            self._side_k = []
            self._side_h = []
            return None

        dets = [d for _s, d, _f in pool]
        frames = [f for _s, _d, f in pool]

        ratios = np.array([float(d.circle_ratio) for d in dets], dtype=np.float64)
        rfills = np.array([float(d.rect_fill) for d in dets], dtype=np.float64)
        ks = np.array([float(d.k_xy) for d in dets], dtype=np.float64)
        rfill_m = float(np.median(rfills)) if len(rfills) else 0.0
        ratio_disp = float(np.median(ratios)) if len(ratios) else 0.0

        longs = sorted(float(max(d.size_mm)) for d in dets)
        shorts = sorted(float(min(d.size_mm)) for d in dets)
        mid = len(dets) // 2
        size_mm = (longs[mid], shorts[mid])

        # K по постановке: медиана по кадрам из середины ROI + защита от «коробки»
        k_xy = float(np.median(ks)) if len(ks) else 0.0
        rfill_tight = float(np.median([rect_fill_score(d.contour) for d in dets]))
        k_eff = effective_k(k_xy, min(rfill_m, rfill_tight))

        k_side = float(np.median(self._side_k)) if self._side_k else None
        height_mm = float(np.median(self._side_h)) if self._side_h else None
        if k_side is not None:
            # круг хотя бы в одной проекции → D (вопросы №22, №24, №27)
            k_eff = max(k_eff, k_side)

        cat = classify(size_mm, k_eff, height_mm)

        pick_i = 0
        for i, d in enumerate(dets):
            if d.category == cat:
                pick_i = i
                break
        base = dets[pick_i]
        self._final = Detection(
            contour=base.contour,
            bbox=base.bbox,
            size_mm=size_mm,
            circle_ratio=ratio_disp,
            category=cat,
            area_px=base.area_px,
            circle_in=base.circle_in,
            circle_out=base.circle_out,
            rect_fill=rfill_m,
            r_in_out=float(np.median([d.r_in_out for d in dets])),
            k_xy=k_xy,
            k_side=k_side,
            height_mm=height_mm,
        )
        self._final_frame = frames[pick_i]
        self._cands = []
        self._fallback = None
        self._best = None
        self._side_k = []
        self._side_h = []
        print(
            f"[passage] mid={self._mid_obs}/{self._obs} cands={len(dets)} "
            f"K_xy={[round(float(k), 2) for k in ks]} "
            f"rfill={[round(float(r), 2) for r in rfills]} "
            f"→ {ZONE[cat]} K={k_eff:.3f} "
            f"K_side={'—' if k_side is None else round(k_side, 3)} "
            f"H={'—' if height_mm is None else round(height_mm, 1)} "
            f"{size_mm[0]:.0f}×{size_mm[1]:.0f}мм",
            flush=True,
        )
        return self._final


class MotionTracker:
    """Выделение целевого объекта: OpenCV-трек + FastSAM уточнение контура."""

    def __init__(self, use_sam: bool = True) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=160, varThreshold=18, detectShadows=False
        )
        self._prev_gray: np.ndarray | None = None
        self._roi_hw: tuple[int, int] | None = None
        self._cx: float | None = None
        self._cy: float | None = None
        self._vx = 0.0
        self._vy = 0.0
        self._area: float | None = None
        self._last_cnt_roi: np.ndarray | None = None
        self._miss = 0
        self._hold_max = 8
        self.drop: str | None = None
        self.drop_pick: str | None = None
        self._path = "seed"
        self._warmup = 0
        self._use_sam = use_sam
        self._refiner: ContourRefiner | None = get_refiner(enabled=use_sam) if use_sam else None
        self._sam_every = 3
        self._sam_i = 0
        self._sam_cnt: np.ndarray | None = None
        self._sam_score = 0.0
        self._sam_area = 0.0
        self._sam_hold = 0  # кадры удержания последнего удачного SAM
        self._sam_hold_max = 12

    def reset(self) -> None:
        use_sam = self._use_sam
        self.__init__(use_sam=use_sam)

    def _reset_track(self) -> None:
        self._cx = self._cy = self._area = None
        self._last_cnt_roi = None
        self._vx = self._vy = 0.0
        self._sam_cnt = None
        self._sam_score = 0.0
        self._sam_hold = 0

    def _cnt_from_seed(
        self, appearance: np.ndarray, seed: np.ndarray, min_area_px: float
    ) -> np.ndarray | None:
        """Полный силуэт объекта по найденному ядру."""
        sm = cv2.moments(seed)
        if sm["m00"] < 1e-6:
            return None
        sx = float(sm["m10"] / sm["m00"])
        sy = float(sm["m01"] / sm["m00"])
        cnt = _component_at(appearance, sx, sy, min_area_px)
        if cnt is None:
            cnt = seed
        return _assemble_object(appearance, cnt, min_area_px)

    def _maybe_apply_sam(
        self,
        roi: np.ndarray,
        cnt: np.ndarray,
        rh: int,
        rw: int,
        *,
        force: bool = False,
    ) -> np.ndarray:
        """SAM только если уверен; в центре ROI — чаще/каждый кадр."""
        if self._refiner is None:
            return cnt
        guide_area = float(cv2.contourArea(cnt))
        gm = cv2.moments(cnt)
        if gm["m00"] < 1e-6:
            return cnt
        gcx = float(gm["m10"] / gm["m00"])
        gcy = float(gm["m01"] / gm["m00"])
        gate_r = max(22.0, 0.55 * float(np.sqrt(max(guide_area, 1.0))))

        # кэш SAM устарел, если он далеко от текущего кандидата — считаем заново
        stale = True
        if self._sam_cnt is not None:
            cm = cv2.moments(self._sam_cnt)
            if cm["m00"] > 1e-6:
                dx = float(cm["m10"] / cm["m00"]) - gcx
                dy = float(cm["m01"] / cm["m00"]) - gcy
                stale = float(np.hypot(dx, dy)) > gate_r
        self._sam_i += 1
        do_infer = force or stale or (self._sam_i % self._sam_every == 0)

        if do_infer:
            refined = self._refiner.refine_contour(roi, cnt)
            if refined is not None:
                # не прыгаем на маску сильно другой площади относительно трека/гайда
                ref_ok = True
                if self._sam_area > 1 and (
                    refined.area < 0.55 * self._sam_area or refined.area > 1.7 * self._sam_area
                ):
                    # допускаем, если сильно лучше по score и близко к текущему opencv
                    if refined.score < 0.55 or refined.area < 0.5 * guide_area or refined.area > 1.9 * guide_area:
                        ref_ok = False
                if refined.area < 0.5 * guide_area or refined.area > 1.9 * guide_area:
                    ref_ok = False
                if ref_ok and refined.score >= max(0.38, self._sam_score * 0.85):
                    self._sam_cnt = refined.contour
                    self._sam_score = refined.score
                    self._sam_area = refined.area
                    self._sam_hold = self._sam_hold_max
                elif ref_ok and self._sam_cnt is None and refined.score >= 0.42:
                    self._sam_cnt = refined.contour
                    self._sam_score = refined.score
                    self._sam_area = refined.area
                    self._sam_hold = self._sam_hold_max

        if self._sam_cnt is None or self._sam_hold <= 0:
            self._sam_hold = 0
            return cnt

        sm2 = cv2.moments(self._sam_cnt)
        if sm2["m00"] < 1e-6:
            self._sam_cnt = None
            self._sam_hold = 0
            return cnt
        scx = float(sm2["m10"] / sm2["m00"])
        scy = float(sm2["m01"] / sm2["m00"])
        # маска SAM обязана совпадать с текущим кандидатом, а не с прошлой позицией
        if float(np.hypot(scx - gcx, scy - gcy)) > gate_r:
            self._sam_cnt = None
            self._sam_score = 0.0
            self._sam_hold = 0
            return cnt

        self._sam_hold -= 0 if do_infer and self._sam_hold == self._sam_hold_max else 1
        # SAM лучше OpenCV по компактности — иначе не подменяем
        a_cv = guide_area
        a_sam = float(cv2.contourArea(self._sam_cnt))
        if a_sam < 0.45 * a_cv or a_sam > 2.1 * a_cv:
            return cnt
        return self._sam_cnt

    def _ensure_roi(self, rh: int, rw: int) -> None:
        if self._roi_hw != (rh, rw):
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=160, varThreshold=18, detectShadows=False
            )
            self._prev_gray = None
            self._roi_hw = (rh, rw)
            self._warmup = 0
            self._cx = self._cy = self._area = None
            self._last_cnt_roi = None
            self._miss = 0
            self._vx = self._vy = 0.0

    def _motion_mask(self, roi: np.ndarray) -> tuple[np.ndarray, float]:
        # статичный товар не должен «въедаться» в фон
        if self._warmup < 20:
            lr = 0.03
        elif self._cx is not None:
            lr = 0.0
        else:
            lr = 0.008
        fg = self._bg.apply(roi, learningRate=lr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            delta = cv2.absdiff(gray, self._prev_gray)
            # выше порог — меньше щелей модульной ленты
            _, dth = cv2.threshold(delta, 28, 255, cv2.THRESH_BINARY)
            # Sobel по разнице кадров — чётче граница движущегося товара
            dmag = _sobel_magnitude(delta)
            _, sth = cv2.threshold(dmag, 35.0, 255, cv2.THRESH_BINARY)
            sth = sth.astype(np.uint8)
            fg = cv2.bitwise_or(fg, dth)
            fg = cv2.bitwise_or(fg, sth)
        self._prev_gray = gray
        self._warmup += 1

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        # щели ≤10 px; карандаш обычно толще
        fg = cv2.subtract(fg, _thin_gap_mask(fg, horizontal=True, max_thick=10))
        fg = cv2.subtract(fg, _thin_gap_mask(fg, horizontal=False, max_thick=10))
        fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        motion_energy = float(np.mean(fg)) / 255.0
        return fg, motion_energy

    def _pick_mask(self, motion: np.ndarray, appearance: np.ndarray, rh: int, rw: int) -> np.ndarray:
        """Выбор только по appearance; motion лишь усиливает уже видимый товар."""
        pick = appearance.copy()
        if float(np.mean(motion)) > 1.5:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            # НИКОГДА не добавляем сырой motion (щели ленты) без appearance
            pick = cv2.bitwise_or(
                pick, cv2.bitwise_and(appearance, cv2.dilate(motion, k, iterations=1))
            )
        if self._cx is not None and self._cy is not None:
            px = int(np.clip(self._cx + self._vx, 0, rw - 1))
            py = int(np.clip(self._cy + self._vy, 0, rh - 1))
            rad = int(max(36.0, 0.65 * np.sqrt(self._area or 900.0)))
            gate = np.zeros((rh, rw), np.uint8)
            cv2.circle(gate, (px, py), rad, 255, -1)
            pick = cv2.bitwise_or(pick, cv2.bitwise_and(appearance, gate))
        return pick

    def _pick_contour(
        self,
        mask: np.ndarray,
        motion: np.ndarray,
        min_area_px: float,
        rw: int,
        rh: int,
        roi_bgr: np.ndarray | None = None,
    ) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        roi_area = float(rw * rh)
        best: tuple[float, np.ndarray] | None = None
        pred_x = (self._cx + self._vx) if self._cx is not None else None
        pred_y = (self._cy + self._vy) if self._cy is not None else None
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV) if roi_bgr is not None else None
        dist_map = _belt_distance_map(roi_bgr) if roi_bgr is not None else None

        rejects: list[tuple[float, str]] = []

        def reject(area: float, bw: int, bh: int, why: str) -> None:
            if area >= 1000.0:
                rejects.append((area, f"{why} [{bw}x{bh} a={int(area)}]"))

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            x, y, bw, bh = cv2.boundingRect(cnt)
            if area > 0.42 * roi_area:
                reject(area, bw, bh, "площадь > 42% ROI")
                continue
            touches = _edge_touches(x, y, bw, bh, rw, rh)
            if touches >= 3:
                reject(area, bw, bh, "касается 3+ краёв")
                continue
            if touches >= 2 and area > 0.22 * roi_area:
                reject(area, bw, bh, "2 края и крупный")
                continue

            # контраст к ленте: щель/шов однотонны, товар — нет
            mean_d = frac_d = -1.0
            strong = False
            if dist_map is not None and roi_bgr is not None:
                mean_d, frac_d = _blob_contrast(roi_bgr, cnt, dist_map)
                if mean_d < 16.0 or frac_d < 0.28:
                    reject(area, bw, bh, f"контраст d={mean_d:.0f} f={frac_d:.2f}")
                    continue
                strong = mean_d >= 32.0 and frac_d >= 0.55

            min_side = float(min(bw, bh))
            max_side = float(max(bw, bh))
            aspect = max_side / max(1.0, min_side)
            if not strong and _is_belt_gap_px(
                min_side, max_side, area=area, bw=bw, bh=bh, rw=rw, rh=rh, touches=touches
            ):
                reject(area, bw, bh, "похоже на щель ленты")
                continue
            stick = _is_stick_px(
                min_side, max_side, area=area, rw=rw, rh=rh, touches=touches
            )
            area_floor = min_area_px * (0.55 if stick or strong else 1.0)
            if area < area_floor:
                reject(area, bw, bh, "мала площадь")
                continue
            if min_side < 12.0:
                reject(area, bw, bh, "тонкий < 12px")
                continue
            # направляющая/кромка ленты: узкая полоса вдоль края ROI
            if aspect > 4.5 and touches >= 1 and min_side < 0.09 * min(rw, rh):
                reject(area, bw, bh, f"полоса у края ROI aspect={aspect:.1f}")
                continue
            if not stick and not strong:
                if min_side < 18:
                    reject(area, bw, bh, "тонкий < 18px")
                    continue
                if aspect > 5.0:
                    reject(area, bw, bh, f"вытянут aspect={aspect:.1f}")
                    continue
                if min_side < 22 and aspect > 3.2:
                    reject(area, bw, bh, f"тонкий+вытянут aspect={aspect:.1f}")
                    continue

            m = cv2.moments(cnt)
            if m["m00"] < 1e-6:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])

            hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
            solidity = area / hull_area if hull_area > 1 else 0.0
            if solidity < (0.62 if stick else 0.50) and not strong:
                reject(area, bw, bh, f"рыхлый solidity={solidity:.2f}")
                continue
            extent = area / max(1.0, float(bw * bh))
            if extent < (0.40 if stick else 0.22) and not strong:
                reject(area, bw, bh, f"extent={extent:.2f}")
                continue

            blob = np.zeros((rh, rw), np.uint8)
            cv2.drawContours(blob, [cnt], -1, 255, -1)
            motion_frac = float(np.mean(motion[blob > 0])) / 255.0 if motion is not None else 0.0
            pack = 1.0 if stick or aspect < 2.8 else float(np.exp(-0.45 * (aspect - 2.8)))
            score = area * (0.55 + 0.3 * solidity + 0.25 * pack + 0.15 * motion_frac)
            if stick:
                score *= 0.70

            if hsv is not None:
                mean_h, mean_s = cv2.mean(hsv, mask=blob)[:2]
                # зелёная линейка — не цель сортировки
                if 35 <= mean_h <= 95 and mean_s > 50:
                    score *= 0.08

            if pred_x is not None and pred_y is not None:
                dist = float(np.hypot(cx - pred_x, cy - pred_y))
                gate = max(40.0, 0.9 * np.sqrt(self._area or area))
                score *= float(np.exp(-0.5 * (dist / gate) ** 2))
                if self._area is not None:
                    ar = area / max(self._area, 1.0)
                    if ar < 0.35 or ar > 2.8:
                        score *= 0.25

            if best is None or score > best[0]:
                best = (score, cnt)

        if best is None and rejects:
            rejects.sort(key=lambda t: -t[0])
            self.drop_pick = rejects[0][1]
        return None if best is None else best[1]

    def _to_detection(
        self,
        cnt_roi: np.ndarray,
        inv: np.ndarray,
        frame_hw: tuple[int, int],
        mm_per_px: float,
        roi_hw: tuple[int, int] | None = None,
    ) -> Detection:
        h, w = frame_hw
        rh, rw = roi_hw if roi_hw is not None else frame_hw
        # круглость — по сырому контуру (approxPolyDP ломает эллипс шайбы)
        ratio_raw, _, _, _, _, rfill_raw, rin_raw = circle_likeness(cnt_roi, (rh, rw))
        cnt = _smooth_contour(cnt_roi, (rh, rw))
        if len(cnt) < 3:
            cnt = cnt_roi
        area = float(cv2.contourArea(cnt))
        # габарит: для эллипса/круга — оси fitEllipse, иначе minAreaRect
        el_f, el_a = ellipse_fill(cnt)
        use_ellipse_size = (
            el_f >= 0.82
            and el_a <= 2.45
            and rfill_raw <= 0.84
            and ratio_raw >= CIRCLE_RATIO_THRESHOLD
            and len(cnt) >= 5
        )
        if use_ellipse_size:
            (_c, (ma, mb), _ang) = cv2.fitEllipse(cnt.astype(np.float32))
            bw, bh = float(max(ma, mb)), float(min(ma, mb))
        else:
            (_c, (rw_mm, rh_mm), _a) = cv2.minAreaRect(cnt.astype(np.float32))
            bw, bh = float(max(rw_mm, rh_mm)), float(min(rw_mm, rh_mm))
        if bw < 1:
            _x, _y, bw_i, bh_i = cv2.boundingRect(cnt)
            bw, bh = float(bw_i), float(bh_i)
        # стержень: толщина ≈ area/length — без ореола маски
        if bw / max(bh, 1.0) >= 3.8 and area > 10.0:
            bh = min(bh, float(area) / max(bw, 1.0) * 1.10)
        size_mm = (bw * mm_per_px, bh * mm_per_px)
        cnt_abs = cv2.perspectiveTransform(cnt.astype(np.float32), inv).astype(np.int32)

        ratio_roi, _, _, _, _, rfill_roi, rin_roi = circle_likeness(cnt, (rh, rw))
        ratio_abs, c_in, r_in, c_out, r_out, rfill_abs, rin_abs = circle_likeness(cnt_abs, (h, w))
        ratio = max(ratio_raw, ratio_roi, ratio_abs)
        rfill = max(rfill_raw, rfill_roi, rfill_abs)
        rin = max(rin_raw, rin_roi, rin_abs)
        # K по постановке — на выпрямленной проекции XY (перспектива снята)
        k_xy = formal_k(cnt, (rh, rw))
        rfill_xy = rect_fill_score(cnt)
        cat = classify(size_mm, effective_k(k_xy, rfill_xy))
        # круглые — красная обводка = гладкая окружность, не зубцы маски
        if cat == Category.CIRCLE and r_out > 1:
            (ox, oy) = c_out
            cnt_abs = _ellipse_contour(((ox, oy), (2 * r_out, 2 * r_out), 0.0))
        xs, ys = cnt_abs[:, 0, 0], cnt_abs[:, 0, 1]
        bx, by = int(xs.min()), int(ys.min())
        return Detection(
            contour=cnt_abs,
            bbox=(bx, by, int(xs.max() - bx), int(ys.max() - by)),
            size_mm=size_mm,
            circle_ratio=ratio,
            category=cat,
            area_px=area,
            circle_in=((c_in[0], c_in[1]), r_in) if r_in > 0.5 else None,
            circle_out=((c_out[0], c_out[1]), r_out) if r_out > 0.5 else None,
            rect_fill=rfill,
            r_in_out=rin,
            k_xy=k_xy,
        )

    def detect(
        self,
        frame: np.ndarray,
        mm_per_px: float,
        min_area_px: float,
        roi_quad=None,
    ) -> list[Detection]:
        h, w = frame.shape[:2]
        quad_px = quad_to_pixels(h, w, roi_quad)
        roi, inv, rw, rh = warp_roi(frame, quad_px)
        self._ensure_roi(rh, rw)

        motion, _energy = self._motion_mask(roi)
        appearance = _appearance_mask(roi)
        belt_dist = _belt_distance_map(roi)
        pick = self._pick_mask(motion, appearance, rh, rw)
        seed = self._pick_contour(pick, motion, min_area_px, rw, rh, roi_bgr=roi)
        if seed is None:
            seed = self._pick_contour(appearance, motion, min_area_px, rw, rh, roi_bgr=roi)
        held = False
        self.drop = None
        self._path = "seed"
        if seed is not None:
            self.drop_pick = None

        if seed is None and self._last_cnt_roi is not None and self._miss < self._hold_max:
            hd, hf = _blob_contrast(roi, self._last_cnt_roi, belt_dist)
            if hd >= 16.0 and hf >= 0.25:
                self._miss += 1
                cnt = self._last_cnt_roi
                held = True
            else:
                # на месте старого контура уже пустая лента — ищем заново без гейта трека
                self._reset_track()
                seed = self._pick_contour(
                    appearance, motion, min_area_px, rw, rh, roi_bgr=roi
                )
                if seed is None:
                    self._miss += 1
                    self.drop = "нет кандидата (после сброса трека)"
                    return []
                cnt = self._cnt_from_seed(appearance, seed, min_area_px)
                if cnt is None:
                    self.drop = "пустой seed"
                    return []
                self._path = "повторный поиск"
        elif seed is None:
            self._miss += 1
            if self._miss > self._hold_max:
                self._reset_track()
            self.drop = "нет кандидата"
            return []
        else:
            built = self._cnt_from_seed(appearance, seed, min_area_px)
            if built is None:
                self.drop = "пустой seed"
                return []
            cnt = built

        m0 = cv2.moments(cnt)
        if m0["m00"] < 1e-6:
            return []
        cx0 = float(m0["m10"] / m0["m00"])
        cy0 = float(m0["m01"] / m0["m00"])
        # середина ROI — усиливаем SAM
        in_mid = abs(cx0 / max(rw, 1) - 0.5) < 0.22 and abs(cy0 / max(rh, 1) - 0.5) < 0.28

        pre_sam = cnt
        if not held and self._refiner is not None:
            cnt = self._maybe_apply_sam(roi, cnt, rh, rw, force=in_mid)
            # маска SAM обязана сидеть на товаре лучше, а не хуже опорной
            if cnt is not pre_sam:
                pd, pf = _blob_contrast(roi, pre_sam, belt_dist)
                sd, sf = _blob_contrast(roi, cnt, belt_dist)
                if sd < pd * 0.9 or sf < pf - 0.05:
                    cnt = pre_sam
                    self._sam_hold = 0
            # шайба: не даём SAM размыть круглый силуэт
            if contour_roundness(pre_sam) >= 0.75 and contour_roundness(cnt) < 0.62:
                cnt = pre_sam
            # в центре: коробку не округляем SAM'ом; иначе берём круглее
            if in_mid:
                if rect_fill_score(pre_sam) >= 0.86:
                    cnt = pre_sam
                else:
                    r_cv = max(contour_roundness(pre_sam), circle_likeness(pre_sam, (rh, rw))[0])
                    r_sam = max(contour_roundness(cnt), circle_likeness(cnt, (rh, rw))[0])
                    if r_cv > r_sam + 0.08:
                        cnt = pre_sam
        elif held and self._sam_cnt is not None and self._sam_hold > 0:
            cnt = self._sam_cnt
            self._sam_hold -= 1

        # отсев щелей ленты; карандаш (stick) оставляем
        dist_map = belt_dist
        _bx, _by, bbw, bbh = cv2.boundingRect(cnt)
        _ms, _Ms = float(min(bbw, bbh)), float(max(bbw, bbh))
        _area = float(cv2.contourArea(cnt))
        _touches = _edge_touches(_bx, _by, bbw, bbh, rw, rh)
        _md, _fd = _blob_contrast(roi, cnt, dist_map)
        _strong = _md >= 32.0 and _fd >= 0.55
        if not _strong and (
            _is_belt_gap_px(_ms, _Ms, area=_area, bw=bbw, bh=bbh, rw=rw, rh=rh, touches=_touches)
            or _ms < 12.0
            or (
                not _is_stick_px(_ms, _Ms, area=_area, rw=rw, rh=rh, touches=_touches)
                and (_ms < 18 or (_ms < 24 and _Ms / max(1.0, _ms) > 4.0))
            )
        ):
            self._miss += 1
            if self._last_cnt_roi is not None and self._miss < self._hold_max:
                cnt = self._last_cnt_roi
                self._path = f"откат после щели ({bbw}x{bbh})"
            else:
                self.drop = f"щель/тонкий: {bbw}x{bbh} area={int(_area)} t={_touches}"
                return []

        if not held:
            cnt = _snap_contour(roi, cnt, dist_map)
            cnt = _trim_appendages(cnt, (rh, rw))

        # удержанный контур на опустевшей ленте — не объект
        mean_d, frac_d = _blob_contrast(roi, cnt, dist_map)
        if mean_d < 16.0 or frac_d < 0.25:
            _dx, _dy, _dw, _dh = cv2.boundingRect(cnt)
            self.drop = (
                f"низкий контраст: d={mean_d:.0f} f={frac_d:.2f} "
                f"{_dw}x{_dh}@{_dx},{_dy} held={held} src={self._path}"
            )
            self._miss += 1
            if self._miss > 2:
                self._reset_track()
            return []

        m = cv2.moments(cnt)
        if m["m00"] < 1e-6:
            return []
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        area = float(cv2.contourArea(cnt))

        if not held:
            if self._cx is not None:
                self._vx = 0.65 * self._vx + 0.35 * (cx - self._cx)
                self._vy = 0.65 * self._vy + 0.35 * (cy - self._cy)
                self._cx = 0.55 * self._cx + 0.45 * cx
                self._cy = 0.55 * self._cy + 0.45 * cy
                self._area = 0.6 * (self._area or area) + 0.4 * area
            else:
                self._cx, self._cy, self._area = cx, cy, area
            self._last_cnt_roi = cnt
            self._miss = 0

        return [self._to_detection(cnt, inv, (h, w), mm_per_px, roi_hw=(rh, rw))]


def detect_objects(
    frame: np.ndarray,
    mm_per_px: float,
    min_area_px: float,
    roi_quad=None,
    tracker: MotionTracker | None = None,
) -> list[Detection]:
    if tracker is not None:
        return tracker.detect(frame, mm_per_px, min_area_px, roi_quad=roi_quad)

    # one-shot без трека: только отличие от фона рамки
    h, w = frame.shape[:2]
    quad_px = quad_to_pixels(h, w, roi_quad)
    roi, inv, rw, rh = warp_roi(frame, quad_px)
    mask = _appearance_mask(roi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    tmp = MotionTracker()
    cnt = tmp._pick_contour(mask, mask, min_area_px, rw, rh, roi_bgr=roi)
    if cnt is None:
        return []
    return [tmp._to_detection(cnt, inv, (h, w), mm_per_px, roi_hw=(rh, rw))]


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    fps: float,
    mm_per_px: float,
    camera_label: str = "",
    roi_quad=None,
    draw_roi: bool = True,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    quad_px = quad_to_pixels(h, w, roi_quad).astype(np.int32)
    if draw_roi:
        # только контур зоны поиска — без заливки (заливка путала сегментацию визуально)
        cv2.polylines(vis, [quad_px], True, (0, 200, 255), 2, cv2.LINE_AA)

    if detections:
        d = detections[0]
        color = COLOR[d.category]
        cv2.drawContours(vis, [d.contour], -1, color, 3, cv2.LINE_AA)
        # описанная (снаружи) — жёлтая
        if d.circle_out:
            (ox, oy), r_out = d.circle_out
            cv2.circle(vis, (int(ox), int(oy)), max(1, int(r_out)), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (int(ox), int(oy)), 3, (0, 220, 255), -1, cv2.LINE_AA)
        # вписанная — бирюзовая
        if d.circle_in:
            (ix, iy), r_in = d.circle_in
            cv2.circle(vis, (int(ix), int(iy)), max(1, int(r_in)), (220, 200, 40), 2, cv2.LINE_AA)
            cv2.circle(vis, (int(ix), int(iy)), 3, (220, 200, 40), -1, cv2.LINE_AA)

    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV классификация товаров")
    p.add_argument("--source", default="auto", help="auto | индекс | URL")
    p.add_argument("--list", action="store_true", help="список камер")
    p.add_argument("--mm-per-px", type=float, default=0.5)
    p.add_argument("--min-area", type=float, default=2500.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        devices = list_avfoundation_devices()
        if not devices:
            print("ffmpeg не вернул список — пробуй индексы 0..3")
            return
        for d in devices:
            print(f"  [{d.index}] {d.name} ({d.kind})")
        return

    if args.source == "auto":
        source, label = find_iphone_source()
    elif args.source.isdigit():
        source, label = int(args.source), f"Camera {args.source}"
    else:
        source, label = args.source, args.source

    print(f"Камера: {label} → {source}")
    cap = open_capture(source)
    mm_per_px = args.mm_per_px
    tracker = MotionTracker()
    win = "ozon-cv"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        dets = detect_objects(frame, mm_per_px, args.min_area, tracker=tracker)
        vis = draw_overlay(frame, dets, fps, mm_per_px, label)
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            tracker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
