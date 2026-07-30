"""PyBullet-реализация core.backend.Backend. core/fsm.py работает с ней ровно так же,
как будет работать с backends.rig_backend.RigBackend — разницы для FSM нет."""
from __future__ import annotations

from core.backend import CameraId, Frame, GateZone
from core.types import Config
from sim.scene import Scene


class SimBackend:
    def __init__(self, config: Config, gui: bool = False):
        self.config = config
        self.scene = Scene(config, gui=gui)

    def grab_frame(self, camera: CameraId) -> Frame:
        return self.scene.grab_frame(camera)

    def reference_frame(self, camera: CameraId) -> Frame:
        return self.scene.reference_frame(camera)

    def belt_speed(self) -> float:
        """Текущая (не номинальная) скорость ленты — если она остановлена сценарием
        faults, FSM должен увидеть именно это, а не подставное значение из конфига."""
        return self.scene.belt_a.speed_mps

    def set_belt_speed(self, mps: float) -> None:
        self.scene.belt_a.speed_mps = mps

    def actuate(self, zone: GateZone) -> bool:
        return self.scene.fire_gate(zone)

    def home(self) -> None:
        self.scene.home()

    # --- методы поверх Protocol, нужны оркестратору сценария (run.py), не FSM ---
    def spawn_item(self, mesh_path: str) -> int:
        return self.scene.spawn_item(mesh_path)

    def remove_item(self, body_id: int) -> None:
        self.scene.remove_item(body_id)

    def step(self, n: int = 1) -> None:
        self.scene.step(n)

    def disconnect(self) -> None:
        self.scene.disconnect()
