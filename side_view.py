#!/usr/bin/env python3
"""Боковая камера: третье измерение и K по вертикальной проекции.

Постановка требует проверять «круг в сечении» по трём ортогональным проекциям
(XY, XZ, YZ) и все три габарита. Камера сверху даёт XY и две стороны основания;
лежащий цилиндр в ней выглядит прямоугольником. Боковой вид добавляет высоту и
вертикальное сечение, поэтому цилиндр на боку уходит в D, а не в B.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from classify import (
    formal_k,
    quad_to_pixels,
    rect_fill_score,
    warp_roi,
)

# нормализованный ROI бокового вида: полоса с лентой у нижней границы
DEFAULT_SIDE_ROI_QUAD = (
    (0.06, 0.30),
    (0.94, 0.30),
    (0.94, 0.86),
    (0.06, 0.86),
)

# доля высоты ROI снизу, в которой должна лежать точка опоры товара:
# при съёмке ровно с уровня полотна лента проецируется в линию у нижней границы
BASE_BAND = 0.12


@dataclass
class SideMeasure:
    height_mm: float
    length_mm: float
    k_side: float
    rect_fill: float
    area_px: float
    bbox_roi: tuple[int, int, int, int]
    contour_abs: np.ndarray

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox_roi
        return {
            "height_mm": round(self.height_mm, 1),
            "length_mm": round(self.length_mm, 1),
            "k_side": round(self.k_side, 3),
            "rect_fill": round(self.rect_fill, 3),
            "area_px": round(self.area_px, 1),
            "bbox_roi": [x, y, w, h],
        }


class SideView:
    """Силуэт товара сбоку: бегущий фон + движение, высота от точки опоры."""

    def __init__(self) -> None:
        self._bg: np.ndarray | None = None  # медленный фон (float32, BGR)
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=250, varThreshold=16, detectShadows=False
        )
        self._roi_hw: tuple[int, int] | None = None
        self.last: SideMeasure | None = None
        self.drop: str | None = None

    def reset(self) -> None:
        self._bg = None
        self._roi_hw = None
        self.last = None
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=250, varThreshold=16, detectShadows=False
        )

    def _update_bg(self, roi: np.ndarray, busy: np.ndarray) -> np.ndarray:
        """Фон обновляем только там, где нет объекта, иначе он «впитает» товар."""
        cur = roi.astype(np.float32)
        if self._bg is None or self._bg.shape != cur.shape:
            self._bg = cur.copy()
            return self._bg
        free = (busy == 0).astype(np.float32)[:, :, None]
        alpha = 0.06 * free + 0.002 * (1.0 - free)
        self._bg = (1.0 - alpha) * self._bg + alpha * cur
        return self._bg

    def _silhouette(self, roi: np.ndarray) -> np.ndarray:
        """Отличие от накопленного фона. Движение самой ленты в маску не берём."""
        motion = self._mog.apply(roi)
        motion = cv2.morphologyEx(
            motion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        bg = self._update_bg(roi, motion)
        diff = cv2.absdiff(roi.astype(np.float32), bg)
        # различие по цвету устойчивее к тени, чем по яркости
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        med = float(np.median(dist))
        mad = float(np.median(np.abs(dist - med))) + 1e-3
        thr = max(18.0, med + 7.0 * mad)
        mask = (dist > thr).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        return mask

    def measure(
        self,
        frame: np.ndarray,
        mm_per_px: float,
        min_area_px: float,
        roi_quad=None,
    ) -> SideMeasure | None:
        self.drop = None
        h, w = frame.shape[:2]
        quad_px = quad_to_pixels(h, w, roi_quad or DEFAULT_SIDE_ROI_QUAD)
        roi, inv, rw, rh = warp_roi(frame, quad_px)
        self._roi_hw = (rh, rw)

        mask = self._silhouette(roi)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            self.drop = "нет силуэта"
            self.last = None
            return None

        best = None
        best_area = 0.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < min_area_px:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # опора: низ силуэта в полосе ленты (её точная строка зависит от
            # того, где товар лежит по ширине полотна)
            if y + bh < rh * (1.0 - BASE_BAND):
                continue
            if bw >= 0.97 * rw and bh >= 0.9 * rh:
                continue  # тряска/смена освещения на весь кадр
            # полоса вдоль всей ленты — это рисунок движущегося полотна, не товар
            if bw > 0.8 * rw and bh < 0.25 * rh:
                continue
            # силуэт до верхней границы ROI: человек/фон, товар в ROI умещается
            if y <= 1:
                continue
            if area > best_area:
                best, best_area = c, area
        if best is None:
            self.drop = "силуэт не на ленте"
            self.last = None
            return None

        x, y, bw, bh = cv2.boundingRect(best)
        # высота — вертикальный размер силуэта, опора уже проверена
        height_mm = float(bh) * mm_per_px
        length_mm = float(bw) * mm_per_px
        k_side = formal_k(best, (rh, rw))
        rfill = rect_fill_score(best)
        cnt_abs = cv2.perspectiveTransform(best.astype(np.float32), inv).astype(np.int32)
        self.last = SideMeasure(
            height_mm=height_mm,
            length_mm=length_mm,
            k_side=k_side,
            rect_fill=rfill,
            area_px=best_area,
            bbox_roi=(int(x), int(y), int(bw), int(bh)),
            contour_abs=cnt_abs,
        )
        return self.last


def draw_side_overlay(
    frame: np.ndarray,
    measure: SideMeasure | None,
    roi_quad=None,
    mm_per_px: float = 0.5,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    quad = quad_to_pixels(h, w, roi_quad or DEFAULT_SIDE_ROI_QUAD).astype(np.int32)
    cv2.polylines(out, [quad], True, (90, 90, 90), 1, cv2.LINE_AA)
    if measure is None:
        cv2.putText(
            out,
            f"side | {mm_per_px:.3f} mm/px | нет объекта",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return out
    cv2.drawContours(out, [measure.contour_abs], -1, (60, 200, 255), 2, cv2.LINE_AA)
    label = f"H={measure.height_mm:.0f}mm  K_side={measure.k_side:.2f}"
    cv2.putText(
        out, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 255), 2, cv2.LINE_AA
    )
    return out
