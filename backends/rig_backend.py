"""Заглушка под реальный физический стенд. Реализует тот же core.backend.Backend,
что и SimBackend — core/fsm.py не отличит одно от другого. Протокол связи с реальным
железом (стенд собирается параллельно) пока не определён, поэтому здесь только
явные TODO на местах, где нужна интеграция.
"""
from __future__ import annotations

from core.backend import CameraId, Frame, GateZone
from core.types import Config


class RigBackend:
    def __init__(self, config: Config):
        self.config = config
        # TODO: подключение к реальным камерам стенда (сколько их, USB/GigE/RTSP?
        # см. core/backend.py CameraId — сейчас предполагается 3: infeed, gate_c, gate_d).
        # TODO: подключение к контроллеру шиберов (какая шина — Modbus/serial/CAN/GPIO?).
        raise NotImplementedError(
            "RigBackend — заглушка. Нужно решить протокол связи с реальными камерами "
            "и контроллером шиберов на стенде, прежде чем подключать этот backend.")

    def grab_frame(self, camera: CameraId) -> Frame:
        # TODO: снять кадр с реальной камеры (RGB) + датчика глубины/дальномера,
        # привести к core.backend.Frame (depth уже в мм, без нелинейного z-buffer —
        # это специфика симуляции, у реального сенсора его нет).
        raise NotImplementedError

    def reference_frame(self, camera: CameraId) -> Frame:
        # TODO: снять и закэшировать кадр пустой сцены один раз при старте/калибровке
        # стенда (до подачи первого товара), см. sim.scene.Scene.reference_frames.
        raise NotImplementedError

    def belt_speed(self) -> float:
        # TODO: либо константа из паспорта привода ленты, либо показания энкодера.
        return self.config.belt_speed_mps

    def actuate(self, zone: GateZone) -> bool:
        # TODO: команда на соленоидный клапан/привод шибера zone, дождаться сигнала
        # о завершении хода (концевик/энкодер), вернуть True только по факту, не сразу
        # после отправки команды — как и SimBackend.actuate(), это блокирующий вызов.
        raise NotImplementedError

    def home(self) -> None:
        # TODO: команда "оба шибера в исходное" + дождаться подтверждения концевиков.
        raise NotImplementedError
