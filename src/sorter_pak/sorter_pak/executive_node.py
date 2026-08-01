"""executive_node - "исполнительная часть": turns /pak/classification into
actuator commands for the 2 shibers and tracks each item to its final
routing outcome (B / C / D).

Logic mirrors sorter_pak.core.conveyor_sim.ConveyorSimulator._try_shiber
exactly (that module is the validated reference for the timing/fail-safe
rules; this node is the same rules wired to ROS2 topics instead of an
in-process loop):

  - shiber 1 (x = X_SHIBER1) fires iff category == 'C' (gabarit).
  - shiber 2 (x = X_SHIBER2) fires iff category == 'D' (circular) OR the
    classification has not arrived yet / is low-confidence - a round or
    oversized item must never reach B (Q&A #33/#34/#97 zero tolerance).
  - otherwise the item is not touched and reaches the B infeed on its own.

Subscribes /pak/item/position (belt encoder feed) and /pak/classification.
Publishes /pak/shiber1/cmd, /pak/shiber2/cmd (actuator commands) and
/pak/routing_result (final B/C/D outcome per item, for metrics/logging).
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sorter_pak.core import conveyor_sim as csim


class ExecutiveNode(Node):
    def __init__(self):
        super().__init__("executive_node")
        self.declare_parameter("shiber_time_s", csim.SHIBER_ACTUATION_S)
        self.shiber_time = float(self.get_parameter("shiber_time_s").value)

        self.pub_shiber1 = self.create_publisher(String, "/pak/shiber1/cmd", 20)
        self.pub_shiber2 = self.create_publisher(String, "/pak/shiber2/cmd", 20)
        self.pub_result = self.create_publisher(String, "/pak/routing_result", 20)

        self.create_subscription(String, "/pak/item/position", self._on_position, 20)
        self.create_subscription(String, "/pak/classification", self._on_classification, 20)

        self.classification = {}   # item_id -> payload dict
        self.routed = set()
        self.shiber1_busy_until = -1.0
        self.shiber2_busy_until = -1.0
        self.get_logger().info("executive_node ready (2 shibers)")

    def _on_classification(self, msg: String):
        data = json.loads(msg.data)
        self.classification[data["item_id"]] = data

    def _fire(self, publisher, name: str, item_id: int, now: float, zone: str):
        publisher.publish(String(data=json.dumps({
            "item_id": item_id, "fire": True, "shiber": name, "t": now,
        })))
        self.get_logger().info(f"{name}: divert item #{item_id} -> {zone}")

    def _on_position(self, msg: String):
        data = json.loads(msg.data)
        now = data["t"]
        for it in data["items"]:
            item_id, x = it["id"], it["x"]
            if item_id in self.routed:
                continue

            cls = self.classification.get(item_id)
            category = cls["category"] if cls else None

            if x >= csim.X_SHIBER1:
                if category == "C":
                    self._fire(self.pub_shiber1, "shiber1_gabarit", item_id, now, "C")
                    self._report(item_id, "C", now)
                    continue
            if x >= csim.X_SHIBER2:
                # fail-safe: unknown/low-confidence classification by the
                # time the item reaches shiber2 is treated as circular.
                if category is None or category == "D":
                    self._fire(self.pub_shiber2, "shiber2_krug", item_id, now, "D")
                    self._report(item_id, "D", now)
                    continue
            if x >= csim.X_END:
                self._report(item_id, "B", now)
                continue

    def _report(self, item_id: int, zone: str, now: float):
        self.routed.add(item_id)
        self.pub_result.publish(String(data=json.dumps({
            "item_id": item_id, "zone": zone, "t": now,
        })))


def main():
    rclpy.init()
    node = ExecutiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
