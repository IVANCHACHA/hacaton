"""Сборка сцены участка ПАК в PyBullet: конвейер A -> зона детекции (камера infeed) ->
шибер C -> шибер D -> (без выталкивания) дальше в B. B физически представлен своим
конвейером на фиксированном по схеме смещении от A (+7000 вдоль оси, +1500 поперёк);
сама стыковка "конец линии A -> начало B" — на усмотрение команды по ТЗ, здесь это
намеренное упрощение: конвейер B стоит на правильном месте геометрически, но не
соединён отдельным механизмом переноса — это отмечено в известных ограничениях.

Расположение вдоль осей: лента A идёт вдоль +X, ширина по Y (центр Y=0). Шибер C
толкает в +Y (лоток и ролл-кейдж C), шибер D — в -Y (лоток и ролл-кейдж D), чтобы
две ветки не пересекались друг с другом и с продолжением линии A.
"""
from __future__ import annotations

import math

import pybullet as p
import pybullet_data
import trimesh

from core.backend import Frame
from core.types import Config
from sim.cameras import Camera
from sim.conveyor import Conveyor
from sim.gates import Gate
from sim.units import to_m
from sim.vhacd_prep import get_convex_decomposition


class Scene:
    def __init__(self, config: Config, gui: bool = False):
        self.config = config
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(config.physics_dt_s)
        p.loadURDF("plane.urdf")

        self.belt_width = config.belt_width_mm
        self.belt_top = config.belt_top_height_mm
        self.detect_x = 500.0
        self.gate_x = {"C": self.detect_x + config.gate_distance_mm["C"],
                        "D": self.detect_x + config.gate_distance_mm["D"]}
        self.push_sign = {"C": 1.0, "D": -1.0}

        # Пока шибер физически идёт через ленту (выброс + возврат), лента не
        # останавливается и продолжает нести товар дальше по X — лоток и камера
        # подтверждения должны стоять не строго над шибером, а над тем местом, где
        # товар реально окажется к концу цикла. drift — грубая оценка этого сноса.
        stroke_mm = self.belt_width + 450.0
        max_force = self._gate_max_force_n(stroke_mm)
        cycle_time_s = 1.6 * (stroke_mm / 1000.0 / config.gate_max_velocity_mps)  # push+retract с запасом
        drift_mm = config.belt_speed_mps * 1000.0 * cycle_time_s
        self.lane_center_x = {zone: self.gate_x[zone] + drift_mm / 2 for zone in ("C", "D")}
        lane_len_x = max(1200.0, drift_mm * 2.0)
        lane_near = self.belt_width / 2
        lane_far = lane_near + 1300.0
        self.lane_len_y = lane_far - lane_near
        self.lane_center_y = {zone: self.push_sign[zone] * (lane_near + lane_far) / 2 for zone in ("C", "D")}

        # старт ленты заметно раньше зоны детекции: спавним товар с запасом, чтобы он
        # успел осесть на ленту и физически проехать до detect_x, а не "телепортировался"
        # в зону детекции сразу после появления (камера infeed иначе видит его мгновенно).
        self.belt_a_start_x = -400.0
        belt_a_len = max(self.lane_center_x.values()) + lane_len_x / 2 - self.belt_a_start_x
        belt_a_center_x = self.belt_a_start_x + belt_a_len / 2
        self.belt_a = Conveyor(self.client, center_mm=(belt_a_center_x, 0.0),
                                length_mm=belt_a_len, width_mm=self.belt_width,
                                top_height_mm=self.belt_top, axis=(1.0, 0.0),
                                speed_mps=config.belt_speed_mps, name="A")

        self.belt_b = Conveyor(self.client,
                                center_mm=(config.a_to_b_long_offset_mm, config.a_to_b_cross_offset_mm),
                                length_mm=1500.0, width_mm=self.belt_width,
                                top_height_mm=self.belt_top, axis=(1.0, 0.0),
                                speed_mps=config.belt_speed_mps, name="B")

        # Conveyor.drive() подгоняет только скорость ВДОЛЬ ленты — поперечный дрейф
        # (осадка после падения на ленту, микро-асимметрия контакта) ничем не
        # компенсируется, и на ~1.5м пути до первого шибера этого достаточно, чтобы
        # широкий товар (до 450мм) съехал за край. У настоящих конвейеров для этого
        # есть направляющие борта — ставим их на участке ДО шиберов (там, где сама
        # сцепка ещё ничего не толкает вбок и дрейфу неоткуда взяться, кроме шума).
        rail_end_x = self.gate_x["C"] - 350.0
        self._build_side_rails(self.belt_a_start_x + 100.0, rail_end_x)

        for zone in ("C", "D"):
            self._build_lane_and_cage(zone, self.push_sign[zone], lane_len_x)

        max_velocity = config.gate_max_velocity_mps
        for zone in ("C", "D"):
            sign = self.push_sign[zone]
            rest_y = -sign * (self.belt_width / 2 + 120.0)
            gate = Gate(self.client, name=zone,
                        rest_position_mm=(self.gate_x[zone], rest_y, self.belt_top + 150.0),
                        push_axis=(0.0, sign), stroke_mm=stroke_mm,
                        plate_size_mm=(600.0, 30.0, 300.0), max_force_n=max_force,
                        max_velocity_mps=max_velocity)
            setattr(self, f"gate_{zone.lower()}", gate)

        # Камера отвода должна реально покрывать всю площадь лотка (лоток намеренно
        # большой — под неопределённость сноса товара во время хода шибера, см. выше),
        # а не только точку над самим шибером — иначе именно так и получалось:
        # товар физически лежит в лотке, а камера смотрит мимо. Радиус обзора считаем
        # из полудиагонали лотка, высоту камеры — из этого радиуса и её FOV.
        gate_fov_deg = 90.0
        half_diag_mm = math.hypot(lane_len_x / 2, self.lane_len_y / 2)
        gate_cam_h = half_diag_mm / math.tan(math.radians(gate_fov_deg / 2)) * 1.15  # 15% запас
        self.cameras = {
            # infeed уже видна вся ширина ленты (500мм) — не нужен запас "на разброс
            # приземления", как у камер на отводах, поэтому кадр можно взять уже,
            # чтобы даже минимальный товар (10x10мм по ТЗ) занимал заметную долю кадра
            "infeed": Camera(self.client, "infeed",
                              eye_mm=(self.detect_x, 0.0, self.belt_top + 550.0),
                              target_mm=(self.detect_x, 0.0, self.belt_top),
                              up=(1, 0, 0), fov_deg=50.0, noise=config.camera_noise),
            "gate_c": Camera(self.client, "gate_c",
                              eye_mm=(self.lane_center_x["C"], self.lane_center_y["C"], self.belt_top + gate_cam_h),
                              target_mm=(self.lane_center_x["C"], self.lane_center_y["C"], self.belt_top),
                              up=(1, 0, 0), fov_deg=gate_fov_deg, noise=config.camera_noise),
            "gate_d": Camera(self.client, "gate_d",
                              eye_mm=(self.lane_center_x["D"], self.lane_center_y["D"], self.belt_top + gate_cam_h),
                              target_mm=(self.lane_center_x["D"], self.lane_center_y["D"], self.belt_top),
                              up=(1, 0, 0), fov_deg=gate_fov_deg, noise=config.camera_noise),
        }
        self.item_ids: list[int] = []
        self._gates_active = 0
        # реальный кадр пустой сцены с каждой камеры (борта, лоток, рама конвейера —
        # всё уже стоит на месте, товаров ещё нет) — core.vision.area_occupied сравнивает
        # с ним попиксельно, а не с угаданной константой "фон на X мм".
        self.reference_frames = {name: cam.capture() for name, cam in self.cameras.items()}

    def _gate_max_force_n(self, stroke_mm: float) -> float:
        """F = m*a*safety, a из кинематики разгона из покоя s=1/2 a t^2 за бюджет
        времени на ход. Не константа — источник see core.types.Config."""
        m = self.config.item_max_mass_kg
        t = self.config.gate_stroke_time_budget_s
        stroke_m = to_m(stroke_mm)
        accel = 2.0 * stroke_m / (t ** 2)
        return m * accel * self.config.gate_safety_factor

    def _build_side_rails(self, x_start: float, x_end: float) -> None:
        length = x_end - x_start
        if length <= 0:
            return
        center_x = (x_start + x_end) / 2
        rail_h = 300.0
        for sign in (1.0, -1.0):
            rail_y = sign * (self.belt_width / 2 + 15.0)
            self._static_box(center_mm=(center_x, rail_y, self.belt_top + rail_h / 2),
                              size_mm=(length, 15.0, rail_h), rgba=(0.5, 0.5, 0.52, 1))

    def _build_lane_and_cage(self, zone: str, sign: float, lane_len_x: float) -> None:
        x = self.lane_center_x[zone]
        lane_near = self.belt_width / 2
        lane_far = lane_near + 1300.0
        lane_center_y = sign * (lane_near + lane_far) / 2
        lane_len_y = lane_far - lane_near
        self._static_box(center_mm=(x, lane_center_y, self.belt_top - 10.0),
                          size_mm=(lane_len_x, lane_len_y, 20.0), rgba=(0.6, 0.6, 0.6, 1))
        cage_y = sign * (lane_far + 400.0)
        self._static_box(center_mm=(x, cage_y, self.config.roll_cage_size_mm[2] / 2),
                          size_mm=self.config.roll_cage_size_mm, rgba=(0.15, 0.35, 0.65, 0.55))

    def _static_box(self, center_mm, size_mm, rgba) -> int:
        half = [to_m(s) / 2 for s in size_mm]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=list(rgba))
        pos = [to_m(v) for v in center_mm]
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                  baseVisualShapeIndex=vis, basePosition=pos)

    def spawn_item(self, mesh_path: str) -> int:
        spawn_pos_mm = (self.belt_a_start_x + 150.0, 0.0, self.belt_top + 60.0)
        mesh_mm = trimesh.load(mesh_path, force="mesh")
        volume_m3 = abs(mesh_mm.volume) * 1e-9
        mass_kg = min(max(volume_m3 * self.config.item_density_kg_m3, 0.05),
                      self.config.item_max_mass_kg)

        # STL организаторов НЕ центрированы в своём локальном начале координат
        # (типично для CAD-экспорта — начало в углу/грани, не в центре масс). Если
        # спавнить как есть, basePosition окажется у угла товара, а не у его центра,
        # и товар физически перевесит/сползёт с ленты сразу после появления — именно
        # так и было при первом прогоне. Компенсируем это через collisionFramePosition/
        # visualFramePosition: сдвигаем сам меш внутри локального фрейма тела так,
        # чтобы geometric-центр AABB меша совпал с началом координат тела.
        bounds_mm = mesh_mm.bounds
        center_mm = (bounds_mm[0] + bounds_mm[1]) / 2.0
        frame_shift_m = [to_m(-c) for c in center_mm]

        # Коллизия — по выпукло-декомпозированной геометрии (VHACD, см. sim/vhacd_prep.py),
        # не по сырому мешу: один GEOM_MESH для динамического тела Bullet всё равно
        # схлопнет в выпуклую оболочку целиком (грубо неверно для вогнутых товаров —
        # шлем, мешок — и на больших мешах на лету может уронить процесс). Визуал —
        # оригинальный STL, для картинки выпуклая декомпозиция не нужна.
        collision_path = get_convex_decomposition(mesh_path)
        col = p.createCollisionShape(p.GEOM_MESH, fileName=collision_path, meshScale=[0.001, 0.001, 0.001],
                                      collisionFramePosition=frame_shift_m)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[0.001, 0.001, 0.001],
                                   visualFramePosition=frame_shift_m, rgbaColor=[0.9, 0.7, 0.2, 1])
        body_id = p.createMultiBody(baseMass=mass_kg, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis,
                                     basePosition=[to_m(v) for v in spawn_pos_mm])
        p.changeDynamics(body_id, -1, lateralFriction=0.8)
        self.item_ids.append(body_id)
        return body_id

    def remove_item(self, body_id: int) -> None:
        if body_id in self.item_ids:
            self.item_ids.remove(body_id)
        p.removeBody(body_id)

    def step(self, n: int = 1) -> None:
        # Пока хотя бы один шибер в ходе — поперечную рецентровку ленты (см. Conveyor.drive)
        # отключаем: она тянет товар обратно к центру ленты, то есть прямо против
        # толчка шибера, и первая попытка систематически не дотягивала ход до цели
        # именно из-за этого перетягивания, а не из-за реального сопротивления товара.
        center_gain = 0.0 if self._gates_active > 0 else 4.0
        for _ in range(n):
            self.belt_a.drive(self.item_ids, center_gain_hz=center_gain)
            self.belt_b.drive(self.item_ids, center_gain_hz=center_gain)
            p.stepSimulation()

    def grab_frame(self, camera: str) -> Frame:
        return self.cameras[camera].capture()

    def reference_frame(self, camera: str) -> Frame:
        return self.reference_frames[camera]

    def fire_gate(self, zone: str) -> bool:
        gate = getattr(self, f"gate_{zone.lower()}")
        self._gates_active += 1
        try:
            return gate.fire(step_fn=lambda: self.step(1))
        finally:
            self._gates_active -= 1

    def home(self) -> None:
        for zone in ("C", "D"):
            gate = getattr(self, f"gate_{zone.lower()}")
            gate._drive_to(0.0, lambda: self.step(1), timeout_steps=240)

    def disconnect(self) -> None:
        p.disconnect(self.client)
