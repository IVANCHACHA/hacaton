"""Уточнение контура через FastSAM (PyTorch). OpenCV даёт bbox/трек — SAM чистит маску."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "FastSAM-s.pt"


@dataclass
class RefineResult:
    contour: np.ndarray
    score: float
    area: float


class ContourRefiner:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._model = None
        self._device = "cpu"
        self._fail = False
        self._last_ms = 0.0

    @property
    def ready(self) -> bool:
        return self.enabled and not self._fail and self._ensure()

    def _ensure(self) -> bool:
        if self._fail or not self.enabled:
            return False
        if self._model is not None:
            return True
        try:
            import torch
            from ultralytics import FastSAM

            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = FastSAM(str(WEIGHTS if WEIGHTS.is_file() else "FastSAM-s.pt"))
            dummy = np.zeros((160, 160, 3), np.uint8)
            self._model.predict(
                dummy, device=self._device, imgsz=160, conf=0.4, iou=0.9, verbose=False
            )
            print(f"[sam] FastSAM-s ready ({self._device})", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail = True
            print(f"[sam] отключено: {exc}", flush=True)
            return False

    def refine_contour(
        self,
        image_bgr: np.ndarray,
        contour: np.ndarray,
        pad: int = 16,
    ) -> RefineResult | None:
        """Уточняет контур; None если результат сомнительный (лучше оставить OpenCV)."""
        if contour is None or len(contour) < 3 or not self.ready:
            return None
        assert self._model is not None
        h, w = image_bgr.shape[:2]
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 12 or bh < 12:
            return None
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        crop = image_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 24 or cw < 24:
            return None

        guide = np.zeros((ch, cw), np.uint8)
        shifted = contour.copy()
        shifted[:, 0, 0] -= x1
        shifted[:, 0, 1] -= y1
        cv2.drawContours(guide, [shifted], -1, 255, -1)
        if int(guide.max()) == 0:
            return None
        guide_f = guide > 0
        guide_area = float(guide_f.sum()) + 1e-6

        import time

        t0 = time.perf_counter()
        try:
            results = self._model.predict(
                crop,
                device=self._device,
                imgsz=480,
                conf=0.4,
                iou=0.85,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sam] predict fail: {exc}", flush=True)
            return None
        self._last_ms = (time.perf_counter() - t0) * 1000.0

        if not results or results[0].masks is None:
            return None
        masks = results[0].masks.data.detach().cpu().numpy()
        if masks.ndim != 3 or masks.shape[0] == 0:
            return None

        best_i, best_score = -1, -1.0
        for i, m in enumerate(masks):
            if m.shape != (ch, cw):
                m_bin = (
                    cv2.resize(m.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR) > 0.5
                )
            else:
                m_bin = m > 0.5
            area = float(m_bin.sum())
            # не раздуваем и не схлопываем относительно OpenCV-гайда
            if area < 0.45 * guide_area or area > 1.85 * guide_area:
                continue
            inter = float(np.logical_and(m_bin, guide_f).sum())
            uni = float(np.logical_or(m_bin, guide_f).sum()) + 1e-6
            iou = inter / uni
            cover = inter / guide_area
            if iou < 0.28 and cover < 0.40:
                continue
            # компактность маски
            ys, xs = np.where(m_bin)
            if len(xs) < 20:
                continue
            bb = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            extent = area / max(1.0, float(bb))
            if extent < 0.20:
                continue
            score = 0.55 * iou + 0.30 * cover + 0.15 * min(1.0, extent / 0.55)
            if score > best_score:
                best_score, best_i = score, i

        # строго: слабый SAM хуже зубчатого, но стабильного OpenCV
        if best_i < 0 or best_score < 0.38:
            return None

        m = masks[best_i]
        if m.shape != (ch, cw):
            m_bin = (
                cv2.resize(m.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR) > 0.5
            ).astype(np.uint8) * 255
        else:
            m_bin = (m > 0.5).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, k, iterations=1)
        m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, k, iterations=1)
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < 80:
            return None
        hull_a = float(cv2.contourArea(cv2.convexHull(cnt)))
        if hull_a > 1 and area / hull_a < 0.35:
            return None
        cnt = cnt.astype(np.int32)
        cnt[:, 0, 0] += x1
        cnt[:, 0, 1] += y1
        return RefineResult(contour=cnt, score=float(best_score), area=area)


_refiner: ContourRefiner | None = None


def get_refiner(enabled: bool = True) -> ContourRefiner:
    global _refiner
    if _refiner is None:
        _refiner = ContourRefiner(enabled=enabled)
    return _refiner
