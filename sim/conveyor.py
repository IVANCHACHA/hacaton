"""Лента конвейера в PyBullet: штатного примитива "конвейер" в Bullet нет, стандартный
приём — инъекция скорости в контактных точках. Каждый физический шаг: для всех тел,
касающихся ленты, компонента скорости вдоль оси ленты принудительно подгоняется к
скорости ленты, остальные компоненты (вертикальная, поперечная) не трогаются — так
гравитация и боковой толчок шибера продолжают работать как обычно."""
from __future__ import annotations

import pybullet as p

from sim.units import to_m


class Conveyor:
    def __init__(self, client: int, center_mm: tuple[float, float], length_mm: float,
                 width_mm: float, top_height_mm: float, axis: tuple[float, float] = (1.0, 0.0),
                 speed_mps: float = 1.0, name: str = "conveyor"):
        self.client = client
        self.name = name
        self.speed_mps = speed_mps
        norm = (axis[0] ** 2 + axis[1] ** 2) ** 0.5
        self.axis = (axis[0] / norm, axis[1] / norm)
        self.top_z_m = to_m(top_height_mm)

        length_m, width_m = to_m(length_mm), to_m(width_mm)
        cx, cy = to_m(center_mm[0]), to_m(center_mm[1])
        self.center_xy_m = (cx, cy)
        belt_thickness_m = 0.02

        frame_half = [length_m / 2, width_m / 2, self.top_z_m / 2]
        frame_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=frame_half)
        frame_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=frame_half, rgbaColor=[0.55, 0.56, 0.58, 1])
        self.frame_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=frame_col, baseVisualShapeIndex=frame_vis,
            basePosition=[cx, cy, self.top_z_m / 2])

        belt_half = [length_m / 2, width_m / 2, belt_thickness_m / 2]
        belt_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=belt_half)
        belt_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=belt_half, rgbaColor=[0.08, 0.08, 0.09, 1])
        self.belt_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=belt_col, baseVisualShapeIndex=belt_vis,
            basePosition=[cx, cy, self.top_z_m - belt_thickness_m / 2])
        p.changeDynamics(self.belt_id, -1, lateralFriction=1.2)

    def drive(self, item_ids: list[int], center_gain_hz: float = 4.0) -> None:
        """center_gain_hz — мягкая поперечная рецентровка (аналог развала роликов
        настоящей ленты, которые сами сгоняют груз к центру). Без неё компонента
        скорости ПОПЕРЁК оси ленты вообще ничем не корректируется, и любая мелкая
        асимметрия контакта при осадке товара на ленту накапливается в дрейф к краю
        на длинных перегонах — на практике товар съезжал с ленты ещё до первого
        шибера. Сила рецентровки на порядок слабее хода шибера, поэтому толчку
        шибера не мешает (шибер отрабатывает за доли секунды, а не за ~секунду)."""
        ax, ay = self.axis
        # поперечная ось (перпендикулярно направлению ленты, в плоскости XY)
        px, py = -ay, ax
        for item_id in item_ids:
            contacts = p.getContactPoints(bodyA=self.belt_id, bodyB=item_id)
            if not contacts:
                continue
            (vx, vy, vz), ang = p.getBaseVelocity(item_id)
            along = vx * ax + vy * ay
            correction = self.speed_mps - along
            vx += correction * ax
            vy += correction * ay

            (x, y, _z), _orn = p.getBasePositionAndOrientation(item_id)
            cx, cy = self.center_xy_m
            offset_lateral = (x - cx) * px + (y - cy) * py
            vx -= center_gain_hz * offset_lateral * px
            vy -= center_gain_hz * offset_lateral * py

            p.resetBaseVelocity(item_id, linearVelocity=[vx, vy, vz], angularVelocity=ang)
