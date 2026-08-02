"""perception_node - "часть определения и классификации товара".

Two independent input paths, both ending in the same /pak/classification
signal (so executive_node never needs to know which track produced it):

  1. Synthetic-sensor track (conveyor_sim_node.py, launch/sim_launch.py):
     /pak/sensor/capture carries {item_id, type_name, yaw, t} - the
     "ground truth" a detector would have extracted - and this node runs
     sorter_pak.core.camera_model.classify_from_sensors (2 sensors with
     modelled noise) on the matching catalog entry.

  2. Real-camera track (gz_bridge_node.py, launch/gz_launch.py): this node
     subscribes directly to the two RGB camera image topics bridged from
     Gazebo (camera_top/image, camera_side/image) and runs the actual
     pixel-based pipeline in sorter_pak.core.vision - an appearance/
     contour-based approach adapted from https://github.com/irwkc/robozone
     (a working real-camera classifier for this exact track) - no ground
     truth is consulted, the category is read off real rendered images.
     /pak/sensor/trigger only carries {item_id, t} (i.e. "something is at
     the sensor zone right now"), analogous to a photo-eye/encoder trigger
     in a real line - separate from, and not a substitute for, the vision
     measurement itself.

Either way, the final B/C/D decision is made by the single shared rule in
sorter_pak.core.geometry.decide_category, and the result is published on
/pak/classification after a simulated inference+comms latency.
"""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from sensor_msgs.msg import Image
    _HAVE_SENSOR_MSGS = True
except ImportError:   # pragma: no cover - sensor_msgs missing only in odd installs
    _HAVE_SENSOR_MSGS = False

from sorter_pak.core import stl_dataset as sd
from sorter_pak.core import camera_model as cam
from sorter_pak.core import conveyor_sim as csim
from sorter_pak.core import geometry as geo
from sorter_pak.core import vision as vis

# How long after a trigger we keep waiting for a post-trigger detection
# from both cameras before fail-safe → D. Must cover: (a) trigger firing
# slightly before the item fully enters the FOV (real-pose poll lag /
# analytic fallback), (b) software-rendered cameras on a no-GPU VM
# publishing at ~5-10 Hz. 0.8 s of *sim* time is plenty at RTF≥0.2 and
# short enough that the next item (pitch 0.7 s) can't steal the window
# when max_items>1.
TRIGGER_CAPTURE_WINDOW_S = 1.5

# A detection older than this relative to "now" is ignored even inside the
# capture window. Kept short so a previous item's silhouette can't be
# reused for the next trigger (item pitch is 0.7 s).
VISION_MAX_AGE_S = 0.6

# Number of startup frames used to calibrate each camera's static
# background (see vision.accumulate_background / vision.segment) before
# any reading is treated as a real item measurement. The belt is
# guaranteed empty for the first X_SENSOR/BELT_SPEED_MPS ~= 2.4s after
# start (item0 has to physically travel there first) and both cameras are
# mounted at that same x - comfortably enough sim-time for a handful of
# frames at any plausible camera update rate.
CALIBRATION_FRAMES = 5


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("catalog_path", "")
        self.declare_parameter("seed", 11)
        self.declare_parameter("latency_min_s", csim.INFERENCE_LATENCY_RANGE_S[0])
        self.declare_parameter("latency_max_s", csim.INFERENCE_LATENCY_RANGE_S[1])
        self.declare_parameter("enable_vision", True)
        # "vision" = cameras only (fail-safe D on miss)
        # "synthetic" = catalog + camera_model via type_name in the trigger
        # "auto" = try vision first, fall back to synthetic if the capture
        #          window expires and the trigger carried type_name
        self.declare_parameter("classification_source", "auto")

        catalog_path = self.get_parameter("catalog_path").value or None
        self.catalog_by_name = {it.name: it for it in sd.get_or_build_catalog(catalog_path)}
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.latency_range = (float(self.get_parameter("latency_min_s").value),
                               float(self.get_parameter("latency_max_s").value))
        self.classification_source = str(
            self.get_parameter("classification_source").value or "auto").lower()

        self.pub = self.create_publisher(String, "/pak/classification", 20)
        self.create_subscription(String, "/pak/sensor/capture", self._on_capture, 20)

        # Latest NON-NONE reading per camera: (VisionReading, ros_time_s).
        # Empty frames deliberately do NOT overwrite these - overwriting
        # with None was the live bug that made nearly every trigger fall
        # through to "no fresh TOP camera reading" fail-safe even though
        # the cameras were publishing at ~8 Hz (the frame under the
        # trigger just happened to segment to nothing for one tick).
        self._top_reading = None
        self._side_reading = None
        self._top_bg = None
        self._side_bg = None
        self._top_bg_frames = 0
        self._side_bg_frames = 0
        # Triggers waiting for a post-trigger detection from both cameras.
        # Each entry: {item_id, sensed_t, opened_at, data}
        self._pending = []

        if _HAVE_SENSOR_MSGS and bool(self.get_parameter("enable_vision").value) \
                and self.classification_source in ("vision", "auto"):
            self.create_subscription(Image, "camera_top/image", self._on_top_image, 10)
            self.create_subscription(Image, "camera_side/image", self._on_side_image, 10)
            self.create_subscription(String, "/pak/sensor/trigger", self._on_trigger, 20)
            self.create_timer(0.05, self._resolve_pending)
            self.get_logger().info(
                f"vision track enabled (camera_top/side RGB images), "
                f"source={self.classification_source}")
        elif self.classification_source == "synthetic":
            self.create_subscription(String, "/pak/sensor/trigger", self._on_trigger_synthetic, 20)
            self.get_logger().info("synthetic classification via /pak/sensor/trigger type_name")

        self.get_logger().info("perception_node ready "
                                f"({len(self.catalog_by_name)} item types loaded)")

    # -- track 1: synthetic sensor model (kinematic conveyor_sim_node) --
    def _on_capture(self, msg: String):
        data = json.loads(msg.data)
        item_type = self.catalog_by_name.get(data["type_name"])
        if item_type is None:
            self.get_logger().warn(f"unknown item type {data['type_name']!r}")
            return
        hull = np.asarray(item_type.hull_local)
        result = cam.classify_from_sensors(hull, data["yaw"], rng=self.rng)
        self._publish_delayed({
            "item_id": data["item_id"],
            "category": result.category,
            "fits_gabarit": result.fits_gabarit,
            "is_circular": result.is_circular,
            "low_confidence": result.low_confidence,
            "dims_mm": list(result.reading.dims_sorted),
            "k_values": result.reading.k_values,
            "sensed_t": data["t"],
            "source": "synthetic",
        })

    # -- track 2: real Gazebo RGB cameras ----------------------------------
    def _on_top_image(self, msg: Image):
        try:
            frame = vis.decode_rgb(bytes(msg.data), msg.width, msg.height)
        except ValueError as e:
            self.get_logger().warn(f"camera_top: {e}")
            return
        if self._top_bg_frames < CALIBRATION_FRAMES:
            self._top_bg = vis.accumulate_background(
                self._top_bg, frame, n_prev=self._top_bg_frames)
            self._top_bg_frames += 1
            if self._top_bg_frames == CALIBRATION_FRAMES:
                self.get_logger().info("camera_top: background calibrated")
            return
        reading = vis.measure_top(frame, self._top_bg)
        if reading is not None:
            self._top_reading = (reading, self._now())
            self._resolve_pending()

    def _on_side_image(self, msg: Image):
        try:
            frame = vis.decode_rgb(bytes(msg.data), msg.width, msg.height)
        except ValueError as e:
            self.get_logger().warn(f"camera_side: {e}")
            return
        if self._side_bg_frames < CALIBRATION_FRAMES:
            self._side_bg = vis.accumulate_background(
                self._side_bg, frame, n_prev=self._side_bg_frames)
            self._side_bg_frames += 1
            if self._side_bg_frames == CALIBRATION_FRAMES:
                self.get_logger().info("camera_side: background calibrated")
            return
        reading = vis.measure_side(frame, self._side_bg)
        if reading is not None:
            self._side_reading = (reading, self._now())
            self._resolve_pending()

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_trigger_synthetic(self, msg: String):
        data = json.loads(msg.data)
        self._classify_synthetic(data)

    def _classify_synthetic(self, data):
        type_name = data.get("type_name")
        item_type = self.catalog_by_name.get(type_name) if type_name else None
        if item_type is None:
            self.get_logger().warn(
                f"item #{data.get('item_id')}: synthetic fallback requested but "
                f"type_name={type_name!r} missing/unknown - failing safe to D")
            self._publish_delayed(self._fail_safe_payload(data, "synthetic_no_type"))
            return
        hull = np.asarray(item_type.hull_local)
        yaw = float(data.get("yaw", 0.0))
        result = cam.classify_from_sensors(hull, yaw, rng=self.rng)
        self.get_logger().info(
            f"item #{data['item_id']}: synthetic -> {result.category} "
            f"(type={type_name})")
        self._publish_delayed({
            "item_id": data["item_id"],
            "category": result.category,
            "fits_gabarit": result.fits_gabarit,
            "is_circular": result.is_circular,
            "low_confidence": result.low_confidence,
            "dims_mm": list(result.reading.dims_sorted),
            "k_values": result.reading.k_values,
            "sensed_t": data["t"],
            "source": "synthetic",
        })

    def _on_trigger(self, msg: String):
        data = json.loads(msg.data)
        if self.classification_source == "synthetic":
            self._classify_synthetic(data)
            return
        self._pending.append({
            "item_id": data["item_id"],
            "sensed_t": data["t"],
            "opened_at": self._now(),
            "data": data,
        })
        self.get_logger().info(
            f"item #{data['item_id']}: capture window opened "
            f"({TRIGGER_CAPTURE_WINDOW_S:.2f}s sim)")
        self._resolve_pending()

    def _fresh_pair(self, opened_at: float):
        """Return (top_r, side_r) if both cameras have a non-None reading
        still within VISION_MAX_AGE_S of now.

        Readings taken *before* the trigger are deliberately accepted: the
        item usually enters the FOV a fraction of a second before x crosses
        X_SENSOR, so the pre-trigger silhouette is the correct one. Cross-
        item reuse is prevented by (a) the short VISION_MAX_AGE_S relative
        to the 0.7 s item pitch and (b) clearing both readings after a
        successful classify. `opened_at` is kept for the capture-window
        timeout in `_resolve_pending`, not for filtering readings here.
        """
        del opened_at  # documented above - used by caller for the window only
        now = self._now()
        top = self._top_reading
        side = self._side_reading
        if top is None or side is None:
            return None
        top_r, top_t = top
        side_r, side_t = side
        if top_r is None or side_r is None:
            return None
        if (now - top_t) > VISION_MAX_AGE_S or (now - side_t) > VISION_MAX_AGE_S:
            return None
        return top_r, side_r

    def _resolve_pending(self):
        if not self._pending:
            return
        now = self._now()
        still = []
        for pend in self._pending:
            pair = self._fresh_pair(pend["opened_at"])
            if pair is not None:
                self._classify_vision(pend, pair[0], pair[1])
                # Consume the readings so the next item can't reuse them.
                self._top_reading = None
                self._side_reading = None
                continue
            if (now - pend["opened_at"]) > TRIGGER_CAPTURE_WINDOW_S:
                top_age = (now - self._top_reading[1]) if self._top_reading else None
                side_age = (now - self._side_reading[1]) if self._side_reading else None
                data = pend["data"]
                if (self.classification_source == "auto"
                        and data.get("type_name") in self.catalog_by_name):
                    self.get_logger().warn(
                        f"item #{pend['item_id']}: vision window miss "
                        f"(top={'age='+f'{top_age:.2f}' if top_age is not None else 'none'}, "
                        f"side={'age='+f'{side_age:.2f}' if side_age is not None else 'none'}) "
                        "- falling back to synthetic")
                    self._classify_synthetic(data)
                else:
                    self.get_logger().warn(
                        f"item #{pend['item_id']}: no TOP+SIDE detection within "
                        f"{TRIGGER_CAPTURE_WINDOW_S:.2f}s capture window "
                        f"(top={'age='+f'{top_age:.2f}' if top_age is not None else 'none'}, "
                        f"side={'age='+f'{side_age:.2f}' if side_age is not None else 'none'}) "
                        "- failing safe to D")
                    self._publish_delayed(self._fail_safe_payload(
                        data, "no_fresh_pair_in_window"))
                continue
            still.append(pend)
        self._pending = still

    def _classify_vision(self, pend, top_r, side_r):
        # height_mm comes from the SIDE camera (silhouette bounding-box
        # height), not TOP - a single top-down RGB view has no depth
        # channel to measure height from at all (see core/vision.py).
        # That height also corrects the TOP camera's own mm-per-pixel
        # scale (vis.top_mm_per_px).
        mm_per_px = vis.top_mm_per_px(side_r.height_mm or 0.0)
        length_mm = top_r.length_px * mm_per_px
        width_mm = top_r.width_px * mm_per_px
        dims_sorted = tuple(sorted(
            [length_mm, width_mm, side_r.height_mm], reverse=True))
        k_values = {"top": top_r.k, "side": side_r.k}
        category, fits, circular, low_conf = geo.decide_category(
            dims_sorted, k_values.values())

        self.get_logger().info(
            f"item #{pend['item_id']}: vision -> {category} "
            f"dims_mm={[round(d, 1) for d in dims_sorted]} "
            f"K_top={top_r.k:.3f} K_side={side_r.k:.3f}")
        self._publish_delayed({
            "item_id": pend["item_id"],
            "category": category,
            "fits_gabarit": fits,
            "is_circular": circular,
            "low_confidence": low_conf,
            "dims_mm": list(dims_sorted),
            "k_values": k_values,
            "sensed_t": pend["sensed_t"],
            "source": "vision",
        })

    def _fail_safe_payload(self, trigger_data, reason):
        return {
            "item_id": trigger_data["item_id"], "category": "D",
            "fits_gabarit": True, "is_circular": True, "low_confidence": True,
            "dims_mm": [], "k_values": {}, "sensed_t": trigger_data["t"],
            "source": "vision", "fail_safe_reason": reason,
        }

    # -- shared -------------------------------------------------------------
    def _publish_delayed(self, payload):
        latency = float(self.rng.uniform(*self.latency_range))

        def publish():
            self.pub.publish(String(data=json.dumps(payload)))
            timer.cancel()

        timer = self.create_timer(latency, publish)


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
