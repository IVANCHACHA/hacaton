"""Конечный автомat одного товара: IDLE -> DETECT -> CLASSIFY -> TRACK -> ACTUATE ->
CONFIRM -> DONE, с таймаутом на каждом состоянии и веткой RECOVERY при срыве
актуации шибера или таймауте. Несколько ItemFSM работают параллельно (каждый товар —
свой экземпляр), это и моделирует несколько товаров в полёте одновременно.

Изначальный набросок ТЗ описывал цикл под руку с присоской (APPROACH/DESCEND/GRIP/
LIFT/TRANSFER/RELEASE) — исполнительная часть реального стенда это 2 шибера, поэтому
цикл пересобран под них: TRACK ждёт расчётного времени подлёта товара к нужному
шиберу (упреждение = расстояние/скорость ленты), ACTUATE — сам импульс шибера.
`actuate()` в Backend по контракту включает в себя и выброс штока, и возврат в
состояние покоя одним циклом — поэтому отдельного состояния RESET в самом FSM нет,
возврат механизма проверяется на уровне backend/метрик участка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import trimesh

from core.backend import Backend
from core.types import Category, ClassificationResult, Config
from core.vision import area_occupied


class State(str, Enum):
    IDLE = "IDLE"
    DETECT = "DETECT"
    CLASSIFY = "CLASSIFY"
    TRACK = "TRACK"
    ACTUATE = "ACTUATE"
    CONFIRM = "CONFIRM"
    RECOVERY = "RECOVERY"
    DONE = "DONE"


@dataclass
class Timeouts:
    DETECT: float = 2.5
    CLASSIFY: float = 0.3
    TRACK: float = 6.0
    CONFIRM: float = 0.8
    RECOVERY: float = 0.2


@dataclass
class ItemLog:
    item_id: str
    source_file: str
    obb_mm: tuple[float, float, float] | None = None
    k_value: float | None = None
    category_pred: str | None = None
    category_true: str | None = None
    zone_actual: str | None = None
    t_detect: float | None = None
    t_classify: float | None = None
    t_actuate: float | None = None
    t_confirm: float | None = None
    t_cycle: float | None = None
    grip_attempts: int = 0  # число импульсов шибера на этот товар
    success: bool = False
    notes: str = ""


ClassifyFn = Callable[[trimesh.Trimesh, Config], ClassificationResult]


class ItemFSM:
    def __init__(self, item_id: str, source_file: str, mesh: trimesh.Trimesh,
                 config: Config, backend: Backend, classify_fn: ClassifyFn,
                 timeouts: Timeouts | None = None, category_true: str | None = None):
        self.item_id = item_id
        self.mesh = mesh
        self.config = config
        self.backend = backend
        self.classify_fn = classify_fn
        self.timeouts = timeouts or Timeouts()
        self.state = State.IDLE
        self.t_in_state = 0.0
        self.t_total = 0.0
        self.log = ItemLog(item_id=item_id, source_file=source_file, category_true=category_true)
        self.target_zone: str | None = None
        self._recovery_attempts = 0

    def start(self) -> None:
        self._goto(State.DETECT)

    @property
    def finished(self) -> bool:
        return self.state == State.DONE

    def _goto(self, state: State) -> None:
        self.state = state
        self.t_in_state = 0.0

    def _timed_out(self, name: str) -> bool:
        return self.t_in_state > getattr(self.timeouts, name)

    def tick(self, dt: float) -> None:
        if self.state == State.DONE:
            return
        self.t_in_state += dt
        self.t_total += dt
        getattr(self, f"_on_{self.state.name.lower()}")()

    def _on_idle(self) -> None:
        pass

    def _on_detect(self) -> None:
        # Ждём, пока товар физически не появится в зоне детекции (по камере infeed) —
        # без этого TRACK считал бы время подлёта к шиберу от старта FSM, а не от
        # реального момента прохождения зоны детекции, и шибер стрелял бы вхолостую
        # ДО подъезда товара (см. известные ограничения: так и было до исправления).
        frame = self.backend.grab_frame("infeed")
        reference = self.backend.reference_frame("infeed")
        seen = frame is not None and area_occupied(
            frame, reference, self.config.chute_occupied_tolerance_mm, self.config.occupancy_min_fraction)
        if not seen:
            if self._timed_out("DETECT"):
                self.log.notes = "таймаут DETECT: товар не появился в зоне детекции"
                return self._goto(State.RECOVERY)
            return
        self.log.t_detect = self.t_total
        self._goto(State.CLASSIFY)

    def _on_classify(self) -> None:
        if self._timed_out("CLASSIFY"):
            self.log.notes = "таймаут CLASSIFY"
            return self._goto(State.RECOVERY)
        result = self.classify_fn(self.mesh, self.config)
        self.log.category_pred = result.category.value
        self.log.k_value = result.k_value
        self.log.obb_mm = result.obb_extents_mm
        self.log.t_classify = self.t_total
        self.target_zone = result.category.value
        self._goto(State.TRACK)

    def _on_track(self) -> None:
        if self.target_zone == Category.B.value:
            # категория B — без выталкивания, товар едет дальше по линии в основной сортировщик
            self.log.zone_actual = "B"
            self.log.success = True
            self.log.t_cycle = self.t_total
            return self._goto(State.DONE)

        gate_distance_mm = self.config.gate_distance_mm.get(self.target_zone)
        if gate_distance_mm is None:
            self.log.notes = f"нет расстояния до шибера для зоны {self.target_zone}"
            return self._goto(State.RECOVERY)
        eta_s = (gate_distance_mm / 1000.0) / max(self.backend.belt_speed(), 1e-6)
        if self.t_in_state >= eta_s:
            return self._goto(State.ACTUATE)
        if self._timed_out("TRACK"):
            self.log.notes = "таймаут TRACK: не дождались подхода товара к шиберу"
            return self._goto(State.RECOVERY)

    def _on_actuate(self) -> None:
        # actuate() по контракту Backend блокирует до конца полного цикла (выброс +
        # возврат штока) и отдаёт только МЕХАНИЧЕСКИЙ итог (клапан сработал, шток дошёл
        # до упора) — однотактное состояние, без собственного таймаута на ожидание.
        # Попал ли товар физически в нужный лоток — отдельно проверяется в CONFIRM.
        self.log.grip_attempts += 1
        mechanically_ok = self.backend.actuate(self.target_zone)  # type: ignore[arg-type]
        self.log.t_actuate = self.t_total
        if mechanically_ok:
            return self._goto(State.CONFIRM)
        self.log.notes = f"шибер {self.target_zone} не завершил ход (застрял/перегружен)"
        self._goto(State.RECOVERY)

    def _on_confirm(self) -> None:
        camera = "gate_c" if self.target_zone == Category.C.value else "gate_d"
        frame = self.backend.grab_frame(camera)  # type: ignore[arg-type]
        reference = self.backend.reference_frame(camera)  # type: ignore[arg-type]
        occupied = frame is not None and area_occupied(
            frame, reference, self.config.chute_occupied_tolerance_mm, self.config.occupancy_min_fraction)
        if occupied:
            self.log.t_confirm = self.t_total
            self.log.zone_actual = self.target_zone
            self.log.success = True
            self.log.t_cycle = self.t_total
            return self._goto(State.DONE)
        if self._timed_out("CONFIRM"):
            self.log.notes = f"камера {camera} не видит товар в лотке после актуации"
            return self._goto(State.RECOVERY)

    def _on_recovery(self) -> None:
        if (self._recovery_attempts < self.config.max_gate_retries
                and self.target_zone and self.target_zone != Category.B.value):
            # повторная попытка импульса шибера (лимит — config.max_gate_retries), пока
            # товар физически ещё в зоне досягаемости — если и они не подтвердятся,
            # дальше уже честный провал, а не бесконечные попытки
            self._recovery_attempts += 1
            self._goto(State.ACTUATE)
            return
        self.log.zone_actual = self.log.zone_actual or "UNKNOWN"
        self.log.success = False
        self.log.t_cycle = self.t_total
        self._goto(State.DONE)
