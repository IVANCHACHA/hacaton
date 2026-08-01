#!/usr/bin/env python3
"""Предложения объектов открытой моделью: FastSAM в режиме «сегментируй всё».

Пороговые маски держатся на подобранных вручную числах и рвутся на текстурной
ленте, на швах полотна и на светлом товаре. FastSAM (ultralytics) режет кадр на
маски без классов — от него нужны границы, а решение «товар или полотно» остаётся
за отличием от фона ленты: шов полотна почти не отличается от ленты, товар — сильно.

Части одного предмета модель часто отдаёт отдельными масками, поэтому годные маски
объединяются, и связные компоненты объединения и есть кандидаты. Отсюда же берётся
несколько объектов в кадре одновременно.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from glare import clipped_mask

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "FastSAM-s.pt"

# те же ворота по отличию от ленты, что и у порогового пути
MIN_DIST = 26.0
MIN_FRAC = 0.42
MAX_AREA_FRAC = 0.42
# модель охотно прихватывает к товару соседнюю выемку полотна: по такому пикселю
# отличие от ленты почти нулевое, поэтому границу маски подрезаем по карте отличий
TRIM_DIST = 16.0
# доля клиппинг-пикселей внутри маски, выше которой это блик, а не товар
MAX_GLARE_FRAC = 0.28


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Дыры внутри силуэта (блик, надпись, тень) — часть товара."""
    inv = cv2.bitwise_not(mask)
    h, w = mask.shape[:2]
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, flood, (0, 0), 0)
    return cv2.bitwise_or(mask, inv)


@dataclass
class Proposal:
    contour: np.ndarray  # в координатах ROI
    area: float
    dist: float  # среднее отличие от ленты внутри маски
    frac: float  # доля пикселей с отличием > 22
    bbox: tuple[int, int, int, int]

    @property
    def strong(self) -> bool:
        return self.dist >= 32.0 and self.frac >= 0.55


class ObjectProposer:
    """Маски FastSAM по всему ROI, отобранные по отличию от полотна."""

    def __init__(self, enabled: bool = True, imgsz: int = 416, conf: float = 0.15) -> None:
        self.enabled = enabled
        # 416 вместо 640: на живом ROI хватает, а инференс заметно быстрее
        self.imgsz = imgsz
        self.conf = conf
        self.last_ms = 0.0
        self.last_n = 0
        self._model = None
        self._device = "cpu"
        self._fail = False

    @property
    def ready(self) -> bool:
        return self.enabled and not self._fail and self._ensure()

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if self._fail or not self.enabled:
            return False
        try:
            import torch
            from ultralytics import FastSAM

            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = FastSAM(str(WEIGHTS if WEIGHTS.is_file() else "FastSAM-s.pt"))
            self._model.predict(
                np.zeros((160, 160, 3), np.uint8),
                device=self._device,
                imgsz=160,
                conf=0.25,
                iou=0.85,
                verbose=False,
            )
            print(f"[prop] FastSAM everything ready ({self._device})", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail = True
            print(f"[prop] отключено: {exc}", flush=True)
            return False

    def _masks(self, roi: np.ndarray) -> list[np.ndarray]:
        assert self._model is not None
        rh, rw = roi.shape[:2]
        t0 = time.perf_counter()
        try:
            res = self._model.predict(
                roi,
                device=self._device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=0.85,
                verbose=False,
                retina_masks=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[prop] predict fail: {exc}", flush=True)
            return []
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        if not res or res[0].masks is None:
            self.last_n = 0
            return []
        raw = res[0].masks.data.detach().cpu().numpy()
        self.last_n = int(raw.shape[0]) if raw.ndim == 3 else 0
        out = []
        for m in raw:
            mb = m > 0.5
            if mb.shape != (rh, rw):
                mb = cv2.resize(m.astype(np.float32), (rw, rh), cv2.INTER_LINEAR) > 0.5
            out.append(mb.astype(np.uint8) * 255)
        return out

    def propose(
        self,
        roi: np.ndarray,
        dist_map: np.ndarray,
        min_area_px: float,
        glare_static: np.ndarray | None = None,
    ) -> list[Proposal]:
        if not self.ready:
            return []
        rh, rw = roi.shape[:2]
        roi_area = float(rh * rw)
        # все текущие засветы (и статичные, и «едущие» блики на глянце) — не товар
        glare = clipped_mask(roi)
        if glare_static is not None and int(glare_static.max()) > 0:
            glare = cv2.bitwise_or(glare, glare_static)
        glare = cv2.dilate(glare, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        evidence = np.zeros((rh, rw), np.uint8)
        for mask in self._masks(roi):
            mask = cv2.subtract(mask, glare)
            area = float(np.count_nonzero(mask))
            if area < 0.35 * min_area_px or area > MAX_AREA_FRAC * roi_area:
                continue
            inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
            pool = dist_map[(inner if np.count_nonzero(inner) > 30 else mask) > 0]
            if pool.size == 0:
                continue
            # полотно и его швы почти не отличаются от фона — так они и отсеиваются
            if float(pool.mean()) < MIN_DIST or float((pool > 22.0).mean()) < MIN_FRAC:
                continue
            evidence = cv2.bitwise_or(evidence, mask)

        if int(evidence.max()) == 0:
            return []
        # кромка ленты (зелёная направляющая) по цвету далека от полотна и лезет
        # в предложения узкой полосой во всю высоту ROI — режем поля, как в маске
        edge = max(3, min(rh, rw) // 40)
        evidence[:, :edge] = 0
        evidence[:, rw - edge :] = 0
        evidence = cv2.subtract(evidence, glare)
        # прихваченное полотно отрезаем, дыры внутри товара оставляем товару
        evidence = cv2.bitwise_and(
            evidence, (dist_map > TRIM_DIST).astype(np.uint8) * 255
        )
        # части одного предмета склеиваем, но не тянемся к соседнему товару
        evidence = cv2.morphologyEx(
            evidence, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        cnts, _ = cv2.findContours(evidence, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        props: list[Proposal] = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < 0.5 * min_area_px or area > MAX_AREA_FRAC * roi_area:
                continue
            solid = np.zeros((rh, rw), np.uint8)
            cv2.drawContours(solid, [c], -1, 255, -1)
            solid = _fill_holes(solid)
            filled, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if filled:
                c = max(filled, key=cv2.contourArea)
                area = float(cv2.contourArea(c))
                solid = np.zeros((rh, rw), np.uint8)
                cv2.drawContours(solid, [c], -1, 255, -1)
            # маска почти целиком из засвета — это блик, не товар
            gfrac = float(np.count_nonzero(cv2.bitwise_and(solid, glare))) / max(area, 1.0)
            if gfrac > MAX_GLARE_FRAC:
                continue
            inner = cv2.erode(solid, np.ones((5, 5), np.uint8))
            pool = dist_map[(inner if np.count_nonzero(inner) > 30 else solid) > 0]
            if pool.size == 0:
                continue
            dist = float(pool.mean())
            frac = float((pool > 22.0).mean())
            if dist < MIN_DIST or frac < MIN_FRAC:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            min_side, max_side = min(bw, bh), max(bw, bh)
            aspect = max_side / max(1.0, float(min_side))
            extent = area / max(1.0, float(bw * bh))
            # полоса / шов модульной ленты (напр. 88×16)
            if min_side < 12 and max_side > 0.55 * max(rw, rh):
                continue
            if aspect >= 4.0 and min_side < 28:
                continue
            if bh < 26 and bw >= int(0.34 * rw):
                continue
            if bw < 26 and bh >= int(0.34 * rh):
                continue
            # зубцы шва посередине полотна
            cx = x + 0.5 * bw
            if abs(cx - 0.5 * rw) < 0.16 * rw and bw < 0.32 * rw:
                hull = float(cv2.contourArea(cv2.convexHull(c)))
                solidity = area / hull if hull > 1 else 0.0
                if solidity < 0.80 or extent < 0.48 or dist < 30.0 or aspect > 2.8:
                    continue
            # мелкое яркое пятно без насыщенности — типичный отблеск фонарика
            patch = roi[y : y + bh, x : x + bw]
            if patch.size and min(bw, bh) < 55:
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                pm = solid[y : y + bh, x : x + bw] > 0
                if pm.any():
                    v_med = float(np.median(hsv[:, :, 2][pm]))
                    s_med = float(np.median(hsv[:, :, 1][pm]))
                    if v_med >= 210 and s_med <= 40 and area < 1.2 * min_area_px:
                        continue
            props.append(
                Proposal(contour=c, area=area, dist=dist, frac=frac, bbox=(x, y, bw, bh))
            )
        props.sort(key=lambda p: -p.area)
        return props


_proposer: ObjectProposer | None = None


def get_proposer(enabled: bool = True) -> ObjectProposer:
    global _proposer
    if _proposer is None:
        _proposer = ObjectProposer(enabled=enabled)
    return _proposer
