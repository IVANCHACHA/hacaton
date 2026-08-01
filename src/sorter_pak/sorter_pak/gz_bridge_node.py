"""gz_bridge_node - drop-in replacement for conveyor_sim_node.py when
running against real Gazebo (gz-sim / Harmonic) physics instead of the
pure-Python kinematic model.

perception_node.py and executive_node.py are used *completely unchanged*
- they only know about `/pak/sensor/capture`, `/pak/item/position`,
`/pak/classification`, `/pak/shiber{1,2}/cmd`, `/pak/routing_result`.
This node's only job is translating between those topics and the
simulated world in worlds/pak_line.sdf:

  - spawns a random catalog item (real STL-derived mesh, see
    models/<slug>/, tools/export_gz_models.py) every ~0.7 s via the
    `ros_gz_sim create` CLI tool, at a random yaw - matching Q&A #1
    ("товар может приезжать в любой ориентации");
  - tracks each item's x-position and republishes it as `/pak/item/position`
    in the exact JSON shape conveyor_sim_node.py already used - see
    "Why time-based position, not the pose bridge" below;
  - fires `/pak/sensor/trigger` ({item_id, t}, NO type/pose - see below)
    once per item when it crosses X_SENSOR;
  - on a shiber "fire" command, gives that specific model a lateral push
    via `gz topic -p ...` on its `/model/<name>/cmd_vel` topic (the
    VelocityControl system baked into every models/<slug>/model.sdf) -
    shiber1 pushes toward roll_cage_C (+Y), shiber2 toward roll_cage_D (-Y);
  - despawns items that have left the monitored segment to keep the
    entity count bounded over a long demo run.

Spawning/removing/pushing are done via the `ros_gz_sim create` and `gz
topic`/`gz service` CLI tools (subprocess) rather than the gz-transport
Python bindings - the CLI surface is stable across gz-sim releases,
whereas the Python binding module names (gz.transportNN) are versioned
and more likely to break silently. See docs/GAZEBO.md if any of these
calls fail on your install.

Deliberately does NOT tell perception_node what type of item this is or
what yaw it was spawned at - that would let classification read the
answer off ground truth instead of the actual rendered RGB cameras
(camera_top/image, camera_side/image - see
sorter_pak/core/vision.py, consumed directly by perception_node.py).
`/pak/sensor/trigger` is only a "look now" signal, analogous to a
photo-eye/encoder trigger on a real line.

Why time-based position, not the pose bridge
----------------------------------------------
The original design bridged gz.msgs.Pose_V (`/world/.../dynamic_pose/info`)
to tf2_msgs/msg/TFMessage and read `child_frame_id` to identify which
transform belongs to which spawned item. On the reference VM this bridge
delivers transforms (confirmed via a call counter - it fires and the
transform count grows with the item count) but BOTH `header.frame_id` and
`child_frame_id` came back as empty strings for every entry - the
Pose_V->TFMessage conversion in this ros_gz_bridge build does not carry
the per-pose `name` field through, so there is no reliable way to match a
transform back to a spawned model name from the ROS2 side of that bridge.

Rather than solve per-model dynamic pose bridging (the same addressing
problem as /model/<name>/cmd_vel, but for a *subscription*, which the
`gz topic -e` one-shot-CLI trick used for commands doesn't have an
equivalent of), this node estimates x analytically: belt speed is fixed
and known (Q&A #87/#98), and the velocity command an item receives at
spawn (and, if diverted, at the shiber) is exactly `self.speed` forward
at all times - so `x(t) = speed * (t - spawn_t)` is exact as long as the
commanded velocity actually landed (see _wait_and_kick/KICK_ATTEMPTS
below for how that's made reliable). This is an open-loop estimate, not
true physics feedback, and is a deliberate, documented simplification -
see docs/GAZEBO.md "Отладка движения".
"""
import json
import os
import subprocess
import threading

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from sorter_pak.core import stl_dataset as sd
from sorter_pak.core import conveyor_sim as csim

WORLD_NAME = "pak_line"
BELT_TOP_Z = 0.5
SPAWN_CLEARANCE_M = 0.006
CLEANUP_AFTER_S = 4.0            # remove an item this long after being routed
DESPAWN_X = csim.X_END + 1.0     # or once this far past the monitored segment

# Belt is 0.5m wide with rails at y=+-0.26 (worlds/pak_line.sdf) - a fully
# random 0-360deg spawn yaw (as originally implemented, matching Q&A #1
# "item can arrive in any orientation") ignores that a wide/square item
# rotated diagonally needs more than 0.5m to clear the rails - e.g. a
# 489x489mm item (Пуфик) needs 489*sqrt(2)=691mm at 45deg. Spawning
# already-interpenetrating the rail collision geometry was observed to
# violently eject the item off the belt entirely. Real operators would
# not place an oversized item diagonally either, so yaw is now restricted
# to whatever range keeps the item's rotated footprint within the rails,
# with the long axis biased fore/aft along the belt (yaw near 0 or pi).
BELT_CLEAR_WIDTH_M = 0.44   # 0.5m rail gap minus a safety margin

# A single `gz topic -p` call is a brand-new, short-lived gz-transport
# process: it has to complete UDP-multicast discovery with the target
# subscriber (the model's VelocityControl plugin) before its one message
# can be delivered - on an idle system this is well under half a second
# (measured), so a couple of spaced-out attempts are already generous
# redundancy. IMPORTANT: keep these counts LOW. Each attempt is a whole
# new OS process (`gz-transport-topic`) doing its own discovery handshake;
# with items spawning every ~0.7s, a double-digit attempt count was
# empirically found to pile up 10-20+ concurrent discovery handshakes at
# once and made delivery *less* reliable, not more (self-inflicted
# congestion) - see docs/GAZEBO.md "Отладка движения" for the full story.
# VelocityControl latches the last command it receives, so a couple of
# identical repeats is harmless, cheap insurance, not the primary fix.
KICK_ATTEMPTS, KICK_INTERVAL_S = 2, 0.4        # not time-critical

# There used to be a separate periodic "rekick" safety net here (re-send
# the forward kick every 0.5s for a while after spawn, to recover an item
# whose initial kick got lost under load). Removed after two things became
# clear from live testing: (1) collision geometry is now a plain box (see
# DIVERT_VY-adjacent history / tools/export_gz_models.py), which made
# items getting genuinely physically stuck a much rarer event than it was
# with convex-hull collision, so the safety net's benefit shrank; (2) every
# extra one-shot `gz topic`/`gz service` CLI call is a brand-new
# short-lived gz-transport connection, and on a long run (100+ items, each
# with several such calls) that connection churn was empirically found to
# eventually crash gz-sim itself (SIGABRT inside libzmq, "Bad address" /
# "Host unreachable" - not fixed by raising the process's open-file limit,
# so it's a gz-transport/ZeroMQ churn issue, not a simple OS resource cap).
# With the safety net's benefit reduced and its cost (one more connection
# per item, always) unchanged, cutting it was the right trade. See
# docs/GAZEBO.md "Длинные прогоны падают" for the full story.
PUSH_ATTEMPTS, PUSH_INTERVAL_S = 1, 0.06       # must land well inside DIVERT_PUSH_S

# Lateral push strong/long enough to actually cross the ~1.0m gap from the
# belt centreline (y=0) to the roll cage centre (y=+-1.0 - see
# worlds/pak_line.sdf roll_cage_C/D, open on the belt-facing side, walled
# on the far side at y=+-1.29 and on the downstream side at
# x=X_SHIBERn+0.29). An earlier attempt (2.5 m/s for SHIBER_ACTUATION_S=
# 0.25s, i.e. only 0.625m) fell short and items ended up on the floor
# beside the cage instead of inside it. Forward motion is now also
# zeroed for the *whole* divert, not just left running: the old code only
# zeroed vy in stop_push() and left vx=self.speed commanded forever,
# so a diverted item kept drifting downstream indefinitely (into the
# cage's side wall and beyond, never settling) instead of coming to rest
# once it reached the cage.
DIVERT_VY = {"shiber1_gabarit": 2.0, "shiber2_krug": -2.0}   # m/s, +Y/-Y
DIVERT_PUSH_S = 0.5    # 2.0 m/s * 0.5s = 1.0m => stops at the cage centre

# Purely visual paddle indicator (worlds/pak_line.sdf shiber1_paddle/
# shiber2_paddle) swung in lockstep with the push above so a live viewer
# can actually see a shiber "fire" - see the world file comment for why
# this carries no physics risk (no <collision>, cannot touch any item).
PADDLE_TOPIC = {"shiber1_gabarit": "/shiber1/paddle_cmd", "shiber2_krug": "/shiber2/paddle_cmd"}
PADDLE_OPEN_ANGLE = {"shiber1_gabarit": 1.1, "shiber2_krug": -1.1}
PADDLE_ATTEMPTS, PADDLE_INTERVAL_S = 1, 0.06   # purely visual - see PUSH_ATTEMPTS comment above
# The paddle is purely decorative, so its open time doesn't need to match
# the real (invisible) push duration (DIVERT_PUSH_S = 0.5s) - give it a
# bit more than that so it's still visibly open when the item settles,
# then closes shortly after.
PADDLE_VISIBLE_S = 0.7


class GzBridgeNode(Node):
    def __init__(self):
        super().__init__("gz_bridge_node")
        self.declare_parameter("catalog_path", "")
        self.declare_parameter("pitch_m", csim.ITEM_PITCH_M)
        self.declare_parameter("belt_speed_mps", csim.BELT_SPEED_MPS)
        self.declare_parameter("seed", 7)
        self.declare_parameter("world", WORLD_NAME)

        self.world = self.get_parameter("world").value
        self.pitch = float(self.get_parameter("pitch_m").value)
        self.speed = float(self.get_parameter("belt_speed_mps").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))

        catalog_path = self.get_parameter("catalog_path").value or None
        self.catalog = sd.get_or_build_catalog(catalog_path)

        # models/ is installed under share/sorter_pak/models (setup.py
        # data_files) - resolved via the ament index, not via __file__
        # hops, since __file__'s behaviour under `colcon build
        # --symlink-install` is not reliable enough to hop back up to the
        # source tree from an installed module.
        models_dir = os.path.join(get_package_share_directory("sorter_pak"), "models")
        slugs_path = os.path.join(models_dir, "slugs.json")
        if not os.path.exists(slugs_path):
            raise RuntimeError(
                f"{slugs_path} not found - run `python3 tools/export_gz_models.py` first "
                "(see docs/GAZEBO.md)")
        with open(slugs_path, encoding="utf-8") as fh:
            self.slug_by_name = json.load(fh)
        self.models_dir = models_dir

        self.pub_sensor = self.create_publisher(String, "/pak/sensor/trigger", 20)
        self.pub_position = self.create_publisher(String, "/pak/item/position", 20)
        self.create_subscription(String, "/pak/shiber1/cmd", self._on_shiber_cmd, 20)
        self.create_subscription(String, "/pak/shiber2/cmd", self._on_shiber_cmd, 20)

        self.next_id = 0
        self.items = {}     # gz_name -> dict(item_id, type_name, x, sensed, spawn_t, routed_t)
        self.next_spawn_t = 0.0
        self.t = 0.0
        self.dt = 0.05
        self.create_timer(self.dt, self._tick)
        self.create_timer(1.0, self._cleanup)
        self.get_logger().info(
            f"gz_bridge_node ready, {len(self.catalog)} item types, world={self.world}")

    # -- spawning ---------------------------------------------------------
    def _safe_yaw(self, dims_mm):
        """Random yaw (rad) that keeps the item's canonical footprint
        (X=longest, Y=mid, in metres) within BELT_CLEAR_WIDTH_M, biased to
        the long axis pointing fore/aft. See BELT_CLEAR_WIDTH_M comment."""
        a, b = dims_mm[0] / 1000.0, dims_mm[1] / 1000.0
        theta_max = 0.0
        for deg in range(1, 91):
            th = np.radians(deg)
            extent = a * abs(np.sin(th)) + b * abs(np.cos(th))
            if extent > BELT_CLEAR_WIDTH_M:
                break
            theta_max = th
        base = float(self.rng.choice([0.0, np.pi]))
        jitter = float(self.rng.uniform(-theta_max, theta_max))
        return base + jitter

    def _spawn(self):
        item_type = self.catalog[self.rng.integers(0, len(self.catalog))]
        slug = self.slug_by_name.get(item_type.name)
        if slug is None:
            return
        gz_name = f"item{self.next_id}_{slug}"
        self.next_id += 1
        yaw = self._safe_yaw(item_type.dims_mm)
        shortest_m = min(item_type.dims_mm) / 1000.0
        spawn_z = BELT_TOP_Z + 0.025 + shortest_m / 2.0 + SPAWN_CLEARANCE_M
        model_file = os.path.join(self.models_dir, slug, "model.sdf")

        cmd = ["ros2", "run", "ros_gz_sim", "create",
               "-world", self.world, "-file", model_file, "-name", gz_name,
               "-x", "0.05", "-y", "0", "-z", f"{spawn_z:.4f}", "-Y", f"{yaw:.4f}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # motion_start_t is when x(t) = speed*(t - motion_start_t) starts
        # being valid - NOT spawn_t: the item sits still at the spawn point
        # until `create` actually finishes and the kick lands, so using
        # spawn_t there would make the predicted x race ahead of the
        # item's true physical position. That mismatch was empirically
        # found to make the sensor trigger fire before the item was
        # actually in the camera's view (-> every classification fell
        # through to the "no fresh camera reading" fail-safe). Set once
        # the kick actually fires, in _wait_and_kick below.
        self.items[gz_name] = {
            "item_id": self.next_id - 1, "type_name": item_type.name, "slug": slug,
            "yaw": yaw, "x": 0.0, "sensed": False, "spawn_t": self.t,
            "motion_start_t": None, "routed_t": None,
        }
        self.get_logger().info(f"spawn {gz_name} ({item_type.name}, yaw={np.degrees(yaw):.0f} deg)")

        # `create` finishing is what actually makes the model (and its
        # VelocityControl plugin) exist - waiting on a *fixed* guessed
        # delay before kicking either fires too early (kick lost, because
        # the model isn't up yet) or, if made safely long, leaves several
        # items queued up motionless at the spawn point at once whenever
        # that delay exceeds the spawn pitch (0.7s) - visible on-screen as
        # a pile-up right at the start of the belt. Waiting on the actual
        # subprocess exit reacts to however long `create` really takes
        # (fast when idle, slower under load) instead of guessing.
        # subprocess.Popen.wait() blocks, so it runs on a background
        # thread, not the rclpy executor thread.
        threading.Thread(target=self._wait_and_kick, args=(gz_name, proc), daemon=True).start()

    def _wait_and_kick(self, gz_name, proc):
        proc.wait()
        it = self.items.get(gz_name)
        if it is None:
            return   # already despawned/cleaned up (racy but harmless if so)
        self._gz_pub_twist_burst(gz_name, self.speed, 0.0, KICK_ATTEMPTS, KICK_INTERVAL_S)
        it["motion_start_t"] = self.t

    # -- shiber commands -> physical push -----------------------------------
    def _on_shiber_cmd(self, msg: String):
        data = json.loads(msg.data)
        if not data.get("fire"):
            return
        shiber = data.get("shiber", "")
        if shiber not in DIVERT_VY:
            return
        gz_name = self._name_for_item(data["item_id"])
        if gz_name is None:
            return
        self.items[gz_name]["routed_t"] = self.t
        self._push(gz_name, shiber)

    def _name_for_item(self, item_id):
        for name, it in self.items.items():
            if it["item_id"] == item_id:
                return name
        return None

    def _gz_pub_burst(self, topic, msgtype, payload, attempts, interval):
        """See KICK_ATTEMPTS/PUSH_ATTEMPTS comment above for why this is a
        burst and not a single `gz topic -p` call."""
        script = (f"for i in $(seq 1 {attempts}); do "
                  f"gz topic -t {topic} -m {msgtype} -p '{payload}'; sleep {interval}; done")
        subprocess.Popen(["bash", "-c", script],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _gz_pub_twist_burst(self, gz_name, vx, vy, attempts, interval):
        topic = f"/model/{gz_name}/cmd_vel"
        payload = f"linear: {{x: {vx}, y: {vy}, z: 0}}"
        self._gz_pub_burst(topic, "gz.msgs.Twist", payload, attempts, interval)

    def _set_paddle(self, shiber, angle):
        topic = PADDLE_TOPIC.get(shiber)
        if topic is None:
            return
        self._gz_pub_burst(topic, "gz.msgs.Double", f"data: {angle}",
                            PADDLE_ATTEMPTS, PADDLE_INTERVAL_S)

    def _push(self, gz_name, shiber):
        # vx=0 (not self.speed): a diverted item must leave the main belt
        # stream, not keep racing downstream while also sliding sideways -
        # see DIVERT_VY comment above for why the old self.speed-forever
        # behaviour left items drifting past the cage instead of settling
        # in it.
        vy = DIVERT_VY[shiber]
        self._gz_pub_twist_burst(gz_name, 0.0, vy, PUSH_ATTEMPTS, PUSH_INTERVAL_S)
        self._set_paddle(shiber, PADDLE_OPEN_ANGLE.get(shiber, 0.0))

        def stop_push():
            self._gz_pub_twist_burst(gz_name, 0.0, 0.0, PUSH_ATTEMPTS, PUSH_INTERVAL_S)
            push_timer.cancel()

        def close_paddle():
            self._set_paddle(shiber, 0.0)
            paddle_timer.cancel()

        push_timer = self.create_timer(DIVERT_PUSH_S, stop_push)
        paddle_timer = self.create_timer(PADDLE_VISIBLE_S, close_paddle)

    # -- housekeeping ------------------------------------------------------
    def _tick(self):
        self.t += self.dt
        if self.t >= self.next_spawn_t:
            self._spawn()
            self.next_spawn_t += self.pitch / self.speed

        for it in self.items.values():
            # analytic position (see module docstring "Why time-based
            # position, not the pose bridge") - valid from motion_start_t
            # (when the velocity kick actually fired), not spawn_t (the
            # item sits still at the spawn point until then).
            if it["motion_start_t"] is None:
                it["x"] = 0.0
                continue
            it["x"] = self.speed * (self.t - it["motion_start_t"])
            if not it["sensed"] and it["x"] >= csim.X_SENSOR:
                it["sensed"] = True
                # deliberately no type_name/yaw here - perception_node
                # reads the real camera_top/camera_side depth images, this
                # is only the "something is here now" trigger.
                self.pub_sensor.publish(String(data=json.dumps({
                    "item_id": it["item_id"], "t": self.t,
                })))

        self.pub_position.publish(String(data=json.dumps({
            "t": round(self.t, 3),
            "items": [{"id": it["item_id"], "x": round(it["x"], 3)}
                      for it in self.items.values()],
        })))

    def _cleanup(self):
        to_remove = []
        for name, it in self.items.items():
            past_end = it["x"] > DESPAWN_X
            routed_stale = it["routed_t"] is not None and (self.t - it["routed_t"]) > CLEANUP_AFTER_S
            if past_end or routed_stale:
                to_remove.append(name)
        for name in to_remove:
            subprocess.Popen(
                ["gz", "service", "-s", f"/world/{self.world}/remove",
                 "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "300", "--req", f'name: "{name}" type: MODEL'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            del self.items[name]


def main():
    rclpy.init()
    node = GzBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
