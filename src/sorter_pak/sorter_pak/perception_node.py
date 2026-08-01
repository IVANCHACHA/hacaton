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

VISION_MAX_AGE_S = 2.5   # ignore a camera reading older than this at trigger time

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

        catalog_path = self.get_parameter("catalog_path").value or None
        self.catalog_by_name = {it.name: it for it in sd.get_or_build_catalog(catalog_path)}
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.latency_range = (float(self.get_parameter("latency_min_s").value),
                               float(self.get_parameter("latency_max_s").value))

        self.pub = self.create_publisher(String, "/pak/classification", 20)
        self.create_subscription(String, "/pak/sensor/capture", self._on_capture, 20)

        self._top_reading = None    # (vis.VisionReading, ros_time_seconds)
        self._side_reading = None
        self._top_bg = None         # per-pixel calibration background (see vision.segment)
        self._side_bg = None
        self._top_bg_frames = 0
        self._side_bg_frames = 0
        if _HAVE_SENSOR_MSGS and bool(self.get_parameter("enable_vision").value):
            self.create_subscription(Image, "camera_top/image", self._on_top_image, 5)
            self.create_subscription(Image, "camera_side/image", self._on_side_image, 5)
            self.create_subscription(String, "/pak/sensor/trigger", self._on_trigger, 20)
            self.get_logger().info("vision track enabled (camera_top/side RGB images)")

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
            self._top_bg = vis.accumulate_background(self._top_bg, frame)
            self._top_bg_frames += 1
            if self._top_bg_frames == CALIBRATION_FRAMES:
                self.get_logger().info("camera_top: background calibrated")
            return
        self._top_reading = (vis.measure_top(frame, self._top_bg), self._now())

    def _on_side_image(self, msg: Image):
        try:
            frame = vis.decode_rgb(bytes(msg.data), msg.width, msg.height)
        except ValueError as e:
            self.get_logger().warn(f"camera_side: {e}")
            return
        if self._side_bg_frames < CALIBRATION_FRAMES:
            self._side_bg = vis.accumulate_background(self._side_bg, frame)
            self._side_bg_frames += 1
            if self._side_bg_frames == CALIBRATION_FRAMES:
                self.get_logger().info("camera_side: background calibrated")
            return
        self._side_reading = (vis.measure_side(frame, self._side_bg), self._now())

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_trigger(self, msg: String):
        data = json.loads(msg.data)
        now = self._now()
        top = self._top_reading
        side = self._side_reading

        if top is None or (now - top[1]) > VISION_MAX_AGE_S or top[0] is None:
            self.get_logger().warn(f"item #{data['item_id']}: no fresh TOP camera reading "
                                    "- failing safe to D")
            self._publish_delayed(self._fail_safe_payload(data, "no_top_reading"))
            return
        if side is None or (now - side[1]) > VISION_MAX_AGE_S or side[0] is None:
            self.get_logger().warn(f"item #{data['item_id']}: no fresh SIDE camera reading "
                                    "- failing safe to D")
            self._publish_delayed(self._fail_safe_payload(data, "no_side_reading"))
            return

        top_r, side_r = top[0], side[0]
        # height_mm comes from the SIDE camera (silhouette bounding-box
        # height), not TOP - a single top-down RGB view has no depth
        # channel to measure height from at all (see core/vision.py
        # module docstring - this is a real, inherent limitation of a
        # plain RGB camera vs the depth camera this pipeline used before).
        # That height also corrects the TOP camera's own mm-per-pixel
        # scale (vis.top_mm_per_px) - without it, tall items' footprint
        # is systematically overstated (observed live: Шлем measured at
        # ~590mm instead of its true 352mm) because the camera is
        # actually imaging the item's top surface, closer to it than the
        # belt the uncorrected scale assumes.
        mm_per_px = vis.top_mm_per_px(side_r.height_mm or 0.0)
        length_mm = top_r.length_px * mm_per_px
        width_mm = top_r.width_px * mm_per_px
        dims_sorted = tuple(sorted(
            [length_mm, width_mm, side_r.height_mm], reverse=True))
        k_values = {"top": top_r.k, "side": side_r.k}
        category, fits, circular, low_conf = geo.decide_category(
            dims_sorted, k_values.values())

        self._publish_delayed({
            "item_id": data["item_id"],
            "category": category,
            "fits_gabarit": fits,
            "is_circular": circular,
            "low_confidence": low_conf,
            "dims_mm": list(dims_sorted),
            "k_values": k_values,
            "sensed_t": data["t"],
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
