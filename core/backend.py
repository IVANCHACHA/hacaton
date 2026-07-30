"""Единый протокол исполнительной части: core/fsm.py работает через него и не знает,
подключён ли backends.sim_backend (PyBullet) или backends.rig_backend (реальный стенд).

Механизм — 2 шибера (поперечных толкателя), не рука с присоской: реальный стенд
использует именно такую исполнительную часть (подтверждено постановкой задачи —
допустимый тип «поперечные толкатели / селекторы»). Товар без команды актуации едет
дальше по линии и физически попадает в зону B без участия backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

GateZone = Literal["C", "D"]
CameraId = Literal["infeed", "gate_c", "gate_d"]


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth_mm: np.ndarray  # (H, W) float32, уже линеаризованная глубина в мм


class Backend(Protocol):
    def grab_frame(self, camera: CameraId) -> Frame:
        """RGB+depth с указанной камеры. Без сегментационной маски — она только для
        оценки ground truth, в пайплайн классификации не подаётся."""
        ...

    def reference_frame(self, camera: CameraId) -> Frame:
        """Кадр пустой сцены (без товара), снятый один раз при старте/калибровке.
        core.vision.area_occupied сравнивает текущий кадр с этим по пикселям — реальное
        фоновое вычитание, а не угаданная константа «пустой фон должен быть на X мм»."""
        ...

    def belt_speed(self) -> float:
        """Текущая скорость ленты, м/с."""
        ...

    def actuate(self, zone: GateZone) -> bool:
        """Блокирующий вызов: полный цикл шибера зоны zone (выброс штока + возврат в
        состояние покоя). Возвращает True, если механизм физически завершил ход
        (клапан сработал, шток дошёл до упора), False — если сорвался/застрял (например,
        слишком тяжёлый или скользкий товар — см. sim/gates.py maxForce). Это ТОЛЬКО
        механическое подтверждение хода, не факт попадания товара в нужный лоток — за
        этим отдельно следит core.fsm через камеру подтверждения (grab_frame gate_c/gate_d)."""
        ...

    def home(self) -> None:
        """Оба шибера — в состояние покоя."""
        ...
