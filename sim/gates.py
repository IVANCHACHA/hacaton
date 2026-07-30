"""Шибер — поперечный толкатель на призматическом шарнире. Управляется
POSITION_CONTROL с ограниченным force: это и есть физический аналог "maxForce" —
слишком тяжёлый/скользкий товар просто не даст плите дойти до цели за то же время,
шток застрянет, и fire() вернёт False по-честному, а не по условию в коде.

Известная грабля PyBullet: тонкие/лёгкие объекты на шаге 1/240с при быстром ходе
плиты (как и лента на 1 м/с) могут "проваливаться" сквозь тонкую геометрию — это
тот же эффект туннелирования, что и с лентой, см. README/ограничения.
"""
from __future__ import annotations

import pybullet as p

from sim.units import to_m


class Gate:
    def __init__(self, client: int, name: str, rest_position_mm: tuple[float, float, float],
                 push_axis: tuple[float, float], stroke_mm: float, plate_size_mm: tuple[float, float, float],
                 max_force_n: float, max_velocity_mps: float = 1.5):
        self.client = client
        self.name = name
        norm = (push_axis[0] ** 2 + push_axis[1] ** 2) ** 0.5
        self.axis = (push_axis[0] / norm, push_axis[1] / norm, 0.0)
        self.stroke_m = to_m(stroke_mm)
        self.max_force_n = max_force_n
        self.max_velocity_mps = max_velocity_mps

        plate_half = [to_m(s) / 2 for s in plate_size_mm]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.001, 0.001, 0.001])
        plate_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=plate_half)
        plate_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=plate_half, rgbaColor=[0.75, 0.2, 0.15, 1])

        base_pos = [to_m(v) for v in rest_position_mm]
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=-1,
            basePosition=base_pos,
            linkMasses=[2.0],
            linkCollisionShapeIndices=[plate_col],
            linkVisualShapeIndices=[plate_vis],
            linkPositions=[[0, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_PRISMATIC],
            linkJointAxis=[list(self.axis)],
        )
        self.joint_index = 0
        p.changeDynamics(self.body_id, self.joint_index, jointDamping=0.1)
        p.setJointMotorControl2(self.body_id, self.joint_index, controlMode=p.POSITION_CONTROL,
                                 targetPosition=0.0, force=self.max_force_n, maxVelocity=self.max_velocity_mps)

    def _drive_to(self, target_m: float, step_fn, timeout_steps: int, tol_m: float = 0.003) -> bool:
        p.setJointMotorControl2(self.body_id, self.joint_index, controlMode=p.POSITION_CONTROL,
                                 targetPosition=target_m, force=self.max_force_n,
                                 maxVelocity=self.max_velocity_mps)
        for _ in range(timeout_steps):
            step_fn()
            pos = p.getJointState(self.body_id, self.joint_index)[0]
            if abs(pos - target_m) < tol_m:
                return True
        return False

    def fire(self, step_fn, timeout_steps: int = 240) -> bool:
        """Полный цикл: выброс штока + возврат в состояние покоя. Возвращает True только
        если механизм реально дошёл до конца хода (не просто "мы скомандовали")."""
        pushed = self._drive_to(self.stroke_m, step_fn, timeout_steps)
        retracted = self._drive_to(0.0, step_fn, timeout_steps)
        return pushed and retracted
