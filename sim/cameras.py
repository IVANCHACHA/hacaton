"""RGB-D камеры PyBullet. Глубина из getCameraImage нелинейна (стандартный
OpenGL z-buffer) — линеаризуем перед пересчётом в мм, известная грабля PyBullet.
Маска сегментации (getCameraImage возвращает её "бесплатно") намеренно вынесена в
отдельный метод и НЕ используется в core/vision.py — только для ground truth оценки."""
from __future__ import annotations

import numpy as np
import pybullet as p

from core.backend import Frame
from sim.units import to_m


class Camera:
    def __init__(self, client: int, name: str, eye_mm: tuple[float, float, float],
                 target_mm: tuple[float, float, float], up=(0.0, 1.0, 0.0), fov_deg: float = 60.0,
                 width: int = 160, height: int = 120, near_m: float = 0.05, far_m: float = 3.0,
                 noise: bool = False):
        self.client = client
        self.name = name
        self.width, self.height = width, height
        self.near_m, self.far_m = near_m, far_m
        self.noise = noise
        eye = [to_m(v) for v in eye_mm]
        target = [to_m(v) for v in target_mm]
        self.view_matrix = p.computeViewMatrix(eye, target, up)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov_deg, width / height, near_m, far_m)

    def _render(self):
        return p.getCameraImage(self.width, self.height, self.view_matrix, self.proj_matrix,
                                 renderer=p.ER_TINY_RENDERER)

    def capture(self) -> Frame:
        w, h, rgb_px, depth_buf, _seg = self._render()
        rgb = np.reshape(rgb_px, (h, w, 4))[:, :, :3].astype(np.uint8)
        depth_buf = np.reshape(depth_buf, (h, w)).astype(np.float64)
        # линеаризация нелинейного z-buffer: depth_buf в [0,1] -> метры -> мм
        depth_m = self.far_m * self.near_m / (self.far_m - (self.far_m - self.near_m) * depth_buf)
        depth_mm = (depth_m * 1000.0).astype(np.float32)
        if self.noise:
            rng = np.random.default_rng()
            depth_mm = depth_mm + rng.normal(0, 1.5, depth_mm.shape).astype(np.float32)
            rgb = np.clip(rgb.astype(np.int16) + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)
        return Frame(rgb=rgb, depth_mm=depth_mm)

    def capture_segmentation_mask(self) -> np.ndarray:
        """ТОЛЬКО для оценки ground truth (results.csv/summary.json). Использовать эту
        маску в core/classifier.py или core/vision.py запрещено условиями задачи."""
        w, h, _rgb, _depth, seg = self._render()
        return np.reshape(seg, (h, w))
