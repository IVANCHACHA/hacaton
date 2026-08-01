"""conveyor_sim_node - stand-in for the physical plant (belt + 2 sensors +
2 shiber actuators) while no real hardware/camera feed is connected.

Publishes:
  /pak/sensor/capture   (std_msgs/String, JSON) - fired once per item when
                         it crosses the sensor zone; carries the item type
                         and yaw, i.e. what a real detector would have
                         extracted from the camera+profiler frames.
  /pak/item/position    (std_msgs/String, JSON) - all active items' x
                         position, published every tick; this is the
                         belt's encoder feed in a real system.

Subscribes:
  /pak/shiber1/cmd, /pak/shiber2/cmd (std_msgs/String, JSON
  {"item_id": int, "fire": bool}) - the executive node's actuator
  commands; on fire=true the nearest active item to that shiber is
  removed from the belt (physically swept into the roll cage).

Swap this node for a real driver (camera/profiler subscriber + PLC/GPIO
shiber output) without touching perception_node or executive_node - that
is the point of keeping the ROS graph split this way.
"""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sorter_pak.core import stl_dataset as sd
from sorter_pak.core import conveyor_sim as csim


class ConveyorSimNode(Node):
    def __init__(self):
        super().__init__("conveyor_sim_node")
        self.declare_parameter("catalog_path", "")
        self.declare_parameter("physics_dt", 0.02)
        self.declare_parameter("pitch_m", csim.ITEM_PITCH_M)
        self.declare_parameter("belt_speed_mps", csim.BELT_SPEED_MPS)
        self.declare_parameter("seed", 7)

        catalog_path = self.get_parameter("catalog_path").value or None
        self.catalog = sd.get_or_build_catalog(catalog_path)
        self.catalog_by_name = {it.name: it for it in self.catalog}
        self.get_logger().info(f"loaded {len(self.catalog)} item types")

        self.dt = float(self.get_parameter("physics_dt").value)
        self.pitch = float(self.get_parameter("pitch_m").value)
        self.speed = float(self.get_parameter("belt_speed_mps").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))

        self.pub_sensor = self.create_publisher(String, "/pak/sensor/capture", 20)
        self.pub_position = self.create_publisher(String, "/pak/item/position", 20)

        self.create_subscription(String, "/pak/shiber1/cmd", self._on_shiber_cmd, 20)
        self.create_subscription(String, "/pak/shiber2/cmd", self._on_shiber_cmd, 20)

        self.t = 0.0
        self.next_spawn_t = 0.0
        self.next_id = 0
        self.active = []   # dicts: id, type_name, x, yaw, sensed
        self.timer = self.create_timer(self.dt, self._tick)

    def _spawn(self):
        item_type = self.catalog[self.rng.integers(0, len(self.catalog))]
        yaw = float(self.rng.uniform(0, 2 * np.pi))
        item = {"id": self.next_id, "type_name": item_type.name, "x": 0.0,
                "yaw": yaw, "sensed": False}
        self.next_id += 1
        self.active.append(item)

    def _on_shiber_cmd(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not data.get("fire"):
            return
        item_id = data["item_id"]
        before = len(self.active)
        self.active = [it for it in self.active if it["id"] != item_id]
        if len(self.active) < before:
            self.get_logger().info(f"plant: item #{item_id} swept off belt "
                                    f"by {data.get('shiber', '?')}")

    def _tick(self):
        if self.t >= self.next_spawn_t:
            self._spawn()
            self.next_spawn_t += self.pitch / self.speed

        for item in self.active:
            item["x"] += self.speed * self.dt
            if not item["sensed"] and item["x"] >= csim.X_SENSOR:
                item["sensed"] = True
                self.pub_sensor.publish(String(data=json.dumps({
                    "item_id": item["id"], "type_name": item["type_name"],
                    "yaw": item["yaw"], "t": self.t,
                })))

        # items that travel past the monitored segment without being
        # diverted are physically gone (handed off to B) - stop tracking.
        self.active = [it for it in self.active if it["x"] < csim.X_END + 0.3]

        self.pub_position.publish(String(data=json.dumps({
            "t": round(self.t, 3),
            "items": [{"id": it["id"], "x": round(it["x"], 3)} for it in self.active],
        })))
        self.t += self.dt


def main():
    rclpy.init()
    node = ConveyorSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
