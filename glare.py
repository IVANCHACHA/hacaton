#!/usr/bin/env python3
"""Блики: фонарик телефона и лампы дают зеркальные засветы на глянце и металле.

Засвет ломает всё сразу: на ленте он выглядит отдельным «товаром», на товаре
пробивает дыру в маске и режет контур, из-за чего K = r_вп/R_оп уезжает.

Отличить блик от светлого товара по одному кадру нельзя: белая коробка тоже яркая
и обесцвеченная. Работает поведение — отблеск от неподвижного источника стоит на
месте в кадре, пока лента и товар едут под ним:

* статичный засвет (лампа, окно, фонарик на полотне, кромка рамы) — выжигаем из
  маски и из оценки фона ленты;
* засвет, который едет вместе с товаром — это часть товара: оставляем в маске,
  дыру внутри контура потом закрывает заливка.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# зеркальный засвет на пластике/ленте; ниже — обычный светлый товар
V_CLIP = 228
# блик обесцвечен — сенсор упёрся во все три канала
S_MAX = 0.32
# доля кадров, после которой засвет считаем приклеенным к кадру
HOT_THR = 0.40
# скорость забывания карты засветов (≈12 кадров памяти)
ALPHA = 0.10


@dataclass
class GlareInfo:
    static: np.ndarray  # 0/255 — засветы, приклеенные к кадру
    moving: np.ndarray  # 0/255 — засветы, едущие вместе с товаром
    static_frac: float
    moving_frac: float

    @property
    def strong(self) -> bool:
        return self.static_frac + self.moving_frac > 0.02

    def to_dict(self) -> dict:
        return {
            "static": round(self.static_frac * 100.0, 2),
            "moving": round(self.moving_frac * 100.0, 2),
            "strong": self.strong,
        }


def clipped_mask(bgr: np.ndarray) -> np.ndarray:
    """Пиксели, выбитые в клиппинг и обесцвеченные — кандидаты в блик."""
    blur = cv2.GaussianBlur(bgr, (3, 3), 0)
    b, g, r = blur[:, :, 0], blur[:, :, 1], blur[:, :, 2]
    mx = np.maximum(np.maximum(b, g), r).astype(np.float32)
    mn = np.minimum(np.minimum(b, g), r).astype(np.float32)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    m = ((mx >= V_CLIP) & (sat <= S_MAX)).astype(np.uint8) * 255
    return cv2.morphologyEx(
        m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )


class GlareTracker:
    """Карта засветов по кадрам: что стоит на месте, а что едет с товаром."""

    def __init__(self, alpha: float = ALPHA, hot_thr: float = HOT_THR) -> None:
        self.alpha = alpha
        self.hot_thr = hot_thr
        self._acc: np.ndarray | None = None
        self.last: GlareInfo | None = None

    def reset(self) -> None:
        self._acc = None
        self.last = None

    def update(self, bgr: np.ndarray) -> GlareInfo:
        h, w = bgr.shape[:2]
        clip = clipped_mask(bgr)
        cur = (clip > 0).astype(np.float32)
        if self._acc is None or self._acc.shape != cur.shape:
            # первый кадр: статику ещё не знаем, считаем всё «едущим»
            self._acc = cur.copy()
        else:
            self._acc = (1.0 - self.alpha) * self._acc + self.alpha * cur
        static = ((self._acc > self.hot_thr) & (clip > 0)).astype(np.uint8) * 255
        static = cv2.dilate(
            static, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        static = cv2.bitwise_and(static, cv2.dilate(clip, np.ones((3, 3), np.uint8)))
        moving = cv2.subtract(clip, static)
        moving = cv2.morphologyEx(
            moving, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        px = float(h * w)
        self.last = GlareInfo(
            static=static,
            moving=moving,
            static_frac=float(np.count_nonzero(static)) / px,
            moving_frac=float(np.count_nonzero(moving)) / px,
        )
        return self.last

    def patch_background(self, bgr: np.ndarray) -> np.ndarray:
        """Кадр для оценки фона: статичные засветы дописаны цветом ленты вокруг."""
        gi = self.last
        if gi is None or gi.static_frac <= 0.0:
            return bgr
        return cv2.inpaint(bgr, gi.static, 4, cv2.INPAINT_TELEA)
