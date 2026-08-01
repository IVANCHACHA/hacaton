"""Discrete-time kinematic simulation of the PAK: fixed-speed conveyor A,
2 sensors (top camera + line profiler, see camera_model.py), 2 shibers
(gabarit diverter -> C, circularity diverter -> D) and straight-through
hand-off to the B infeed.

Layout (metres, X=0 at the start of the fixed conveyor A, matches
тз/Постановка_Задача_3 section "Параметры участка"):

    0                2.1   2.4        3.4          4.4          5.6
    |--fixed A (3x0.7m, cannot move)--|--sensor---shiber1(C)---shiber2(D)---> B infeed
                                        zone

  - 0.0-2.1 m : конвейер A, position/geometry fixed per Q&A #8/#10/#117.
  - 2.4 m     : sensor zone (top camera + line profiler, allowed here
                per Q&A #41 - mounted over/after the fixed section).
  - 3.4 m     : shiber 1 - diverts "не подходит по габаритам" -> roll cage C.
  - 4.4 m     : shiber 2 - diverts "не подходит без доупаковки" (круг) -> roll cage D.
  - 5.6 m     : hand-off to the main sorter infeed B (fixed, Q&A #52/#117;
                B accepts the belt's 1 m/s without jamming - Q&A #106/#107).

No physical накопитель-with-stop is modelled: routing happens inline via
the 2 shibers while the belt keeps moving at a constant 1 m/s, which the
task explicitly allows as an alternative to a classic buffer (Q&A #57).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from . import camera_model as cam

BELT_SPEED_MPS = 1.0                 # fixed, cannot be slowed/stopped (Q&A #87/#98)
ITEM_PITCH_M = 0.7                   # one item per ~700mm section (Q&A #58/#63)

X_FIXED_END = 2.1
X_SENSOR = 2.4
X_SHIBER1 = 3.4
X_SHIBER2 = 4.4
X_END = 5.6

SHIBER_ACTUATION_S = 0.25            # swing-out + sweep + return budget
INFERENCE_LATENCY_RANGE_S = (0.04, 0.12)   # detection+classification+comms budget


@dataclass
class SimConfig:
    duration_s: float = 60.0
    seed: int = 7
    belt_speed: float = BELT_SPEED_MPS
    pitch: float = ITEM_PITCH_M
    frame_dt: float = 0.05           # animation snapshot interval
    physics_dt: float = 0.01
    latency_range: tuple = INFERENCE_LATENCY_RANGE_S
    shiber_time: float = SHIBER_ACTUATION_S


@dataclass
class ShiberState:
    x: float
    name: str
    state: str = "IDLE"              # IDLE | ACTUATING
    until: float = -1.0
    n_actuations: int = 0
    busy_time: float = 0.0

    def actuate(self, now: float, duration: float):
        self.state = "ACTUATING"
        self.until = now + duration
        self.n_actuations += 1
        self.busy_time += duration

    def update(self, now: float):
        if self.state == "ACTUATING" and now >= self.until:
            self.state = "IDLE"


@dataclass
class SimItem:
    item_id: int
    type_name: str
    x: float
    yaw: float
    spawn_t: float
    category_true: str
    category_measured: Optional[str] = None
    decision_ready_t: Optional[float] = None
    sensor_t: Optional[float] = None
    routed_zone: Optional[str] = None    # 'B' | 'C' | 'D'
    routed_t: Optional[float] = None
    late_classification: bool = False


@dataclass
class SimResult:
    events: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    items: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def save(self, path: str, include_frames: bool = True):
        payload = {
            "metrics": self.metrics,
            "events": self.events,
            "items": [asdict(i) for i in self.items],
        }
        if include_frames:
            payload["frames"] = self.frames
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


class ConveyorSimulator:
    """Runs one simulated shift of `cfg.duration_s` seconds. Call run()."""

    def __init__(self, catalog: list, config: SimConfig = None):
        self.catalog = catalog
        self.cfg = config or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0.0
        self.next_spawn_t = 0.0
        self.next_item_id = 0
        self.active: list[SimItem] = []
        self.finished: list[SimItem] = []
        self.shiber1 = ShiberState(x=X_SHIBER1, name="shiber1_gabarit")
        self.shiber2 = ShiberState(x=X_SHIBER2, name="shiber2_krug")
        self.result = SimResult()
        self._next_frame_t = 0.0

    # -- helpers -----------------------------------------------------
    def _log(self, ev_type: str, **kw):
        self.result.events.append({"t": round(self.t, 3), "event": ev_type, **kw})

    def _spawn(self):
        item_type = self.catalog[self.rng.integers(0, len(self.catalog))]
        yaw = float(self.rng.uniform(0, 2 * np.pi))
        item = SimItem(item_id=self.next_item_id, type_name=item_type.name, x=0.0,
                        yaw=yaw, spawn_t=self.t, category_true=item_type.category_true)
        self.next_item_id += 1
        self.active.append(item)
        self._log("spawn", item_id=item.item_id, type=item.type_name,
                   yaw_deg=round(np.degrees(yaw), 1))

    def _sensor_capture(self, item: SimItem):
        item_type = next(it for it in self.catalog if it.name == item.type_name)
        hull = np.asarray(item_type.hull_local)
        measured = cam.classify_from_sensors(hull, item.yaw, rng=self.rng)
        latency = float(self.rng.uniform(*self.cfg.latency_range))
        item.category_measured = measured.category
        item.sensor_t = self.t
        item.decision_ready_t = self.t + latency
        self._log("sensor_capture", item_id=item.item_id, category=measured.category,
                   fits_gabarit=measured.fits_gabarit, is_circular=measured.is_circular,
                   low_confidence=measured.low_confidence, latency_ms=round(latency * 1000, 1))

    def _try_shiber(self, item: SimItem, shiber: ShiberState, target_category: str) -> bool:
        """Returns True if the item was diverted at this shiber."""
        if item.decision_ready_t is None or self.t < item.decision_ready_t:
            item.late_classification = True
            item.category_measured = "D"  # fail-safe (Q&A #33/#34/#97)
            self._log("late_classification_fallback", item_id=item.item_id)
        if item.category_measured == target_category:
            shiber.actuate(self.t, self.cfg.shiber_time)
            item.routed_zone = target_category
            item.routed_t = self.t
            self._log("shiber_actuate", shiber=shiber.name, item_id=item.item_id,
                       zone=target_category)
            return True
        return False

    def _record_frame(self):
        self.result.frames.append({
            "t": round(self.t, 3),
            "items": [{"id": it.item_id, "x": round(it.x, 3), "type": it.type_name,
                       "cat": it.category_measured} for it in self.active],
            "shiber1": self.shiber1.state,
            "shiber2": self.shiber2.state,
        })

    # -- main loop -----------------------------------------------------
    def step(self):
        cfg = self.cfg
        spawn_interval = cfg.pitch / cfg.belt_speed
        if self.t >= self.next_spawn_t:
            self._spawn()
            self.next_spawn_t += spawn_interval

        still_active = []
        for item in self.active:
            item.x += cfg.belt_speed * cfg.physics_dt

            if item.sensor_t is None and item.x >= X_SENSOR:
                self._sensor_capture(item)

            routed = False
            if item.x >= X_SHIBER1:
                routed = self._try_shiber(item, self.shiber1, "C")
            if not routed and item.x >= X_SHIBER2:
                routed = self._try_shiber(item, self.shiber2, "D")
            if not routed and item.x >= X_END:
                item.routed_zone = "B"
                item.routed_t = self.t
                self._log("handoff_B", item_id=item.item_id)
                routed = True

            if routed:
                self.finished.append(item)
            else:
                still_active.append(item)
        self.active = still_active

        self.shiber1.update(self.t)
        self.shiber2.update(self.t)

        if self.t >= self._next_frame_t:
            self._record_frame()
            self._next_frame_t += cfg.frame_dt

        self.t += cfg.physics_dt

    def run(self) -> SimResult:
        while self.t < self.cfg.duration_s:
            self.step()
        self.result.items = self.finished + self.active
        self._compute_metrics()
        return self.result

    # -- metrics ---------------------------------------------------------
    def _compute_metrics(self):
        items = self.result.items
        n = len(items)
        routed = [it for it in items if it.routed_zone is not None]
        in_transit = n - len(routed)
        confusion = {}
        unsafe_to_B = 0
        for it in routed:
            key = (it.category_true, it.routed_zone)
            confusion[key] = confusion.get(key, 0) + 1
            if it.routed_zone == "B" and it.category_true != "B":
                unsafe_to_B += 1
        correct = sum(v for k, v in confusion.items() if k[0] == k[1])
        throughput_per_hour = n / self.cfg.duration_s * 3600.0 if self.cfg.duration_s else 0.0
        self.result.metrics = {
            "n_items": n,
            "n_routed": len(routed),
            "n_in_transit_at_end": in_transit,
            "accuracy_vs_reference": round(correct / len(routed), 4) if routed else None,
            "unsafe_items_routed_to_B": unsafe_to_B,
            "confusion_matrix": {f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())},
            "throughput_items_per_hour": round(throughput_per_hour, 1),
            "shiber1_actuations": self.shiber1.n_actuations,
            "shiber1_duty_cycle": round(self.shiber1.busy_time / self.cfg.duration_s, 4),
            "shiber2_actuations": self.shiber2.n_actuations,
            "shiber2_duty_cycle": round(self.shiber2.busy_time / self.cfg.duration_s, 4),
            "late_classifications": sum(1 for it in items if it.late_classification),
            "min_available_window_s": round(self.cfg.pitch / self.cfg.belt_speed, 3),
            "shiber_actuation_s": self.cfg.shiber_time,
        }


def run_simulation(catalog: list, config: SimConfig = None) -> SimResult:
    return ConveyorSimulator(catalog, config).run()
