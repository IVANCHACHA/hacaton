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
  - on a shiber "fire" command, stops belt VelocityControl on that item,
    swings the visible paddle, and slides it into the matching roll cage
    (C=+Y / D=-Y) with a short set_pose path pinned at the shiber X so
    wide items cannot jam on the rail gap;
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
import re
import subprocess
import threading
import time

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

# Verified live (see docs/GAZEBO.md "Затор у точки спавна"/"Реальные
# позиции, не вымышленные"): even with a reliable kick and stable (box)
# collision, an item that's even briefly slow to clear the spawn point
# becomes a physical obstacle for the next one - every item spawns at the
# exact same (x=0.05, y=0), so this compounds into a visible pile-up
# ("товары застревают в начале", user report + confirmed with an actual
# camera screenshot, not just log analysis). KICK_ATTEMPTS/rekick only
# reduce how OFTEN a kick is lost; they can't stop the pile-up once one
# item is genuinely stuck, and a second, separate bug made this worse
# than it looked: gz_bridge_node's OWN idea of where an item is
# (x(t)=speed*(t-motion_start_t), see "Why time-based position" above)
# kept advancing on schedule even for an item that was, in real physics,
# not moving at all - so sensor triggers and shiber diverts fired for
# items that were still sitting at the spawn point (observed live:
# "diverted" items whose real position was still x<0). A background
# thread polls REAL poses via `gz topic -e .../pose/info` (the *raw*
# gz-transport CLI, not the ROS bridge that was found to lose model
# names - same module docstring) and this is now used for three things:
# (1) `_spawn()` waits for the spawn point to be physically clear before
# creating the next item, instead of spawning blindly on a fixed timer;
# (2) `_tick()` uses the real x directly instead of the analytic estimate
# whenever it's available (see _tick); (3) any not-yet-routed item whose
# real x hasn't meaningfully changed in STUCK_TIMEOUT_S - wherever on the
# belt it is, not just at spawn - is force-despawned so it can't block
# forever (Q&A #101 explicitly leaves "what to do with a stuck/jammed
# item" up to the team). The poll interval is a deliberate trade-off:
# frequent polling tracks stuck items and real position faster, but each
# poll is itself a new short-lived gz-transport connection, and this
# project separately found (docs/GAZEBO.md "Длинные прогоны падают") that
# enough connection churn can crash gz-sim outright on a long run -
# actually knowing where items are is worth that cost more than a
# fast-but-fictional-and-eventually-crashing run is.
# Belt travel is kinematic set_pose (y pinned to 0). VelocityControl on the
# belt was observed live to nudge items sideways through rail gaps and dump
# them on the floor ("сломал ленту"). Shiber divert still uses a short
# set_pose slide into the cage (see _push). No cmd_vel kicks / rekicks.
PADDLE_ATTEMPTS, PADDLE_INTERVAL_S = 2, 0.1

# Hard ceiling on any single `gz topic -p` / `gz service` CLI call, and on
# `ros2 run ros_gz_sim create`. Verified live on the reference VM: a
# `gz topic -p` call whose target subscriber never completes discovery
# (e.g. the model was already despawned, or the transport layer is
# congested) blocks INDEFINITELY - there is no built-in timeout on the CLI
# itself. Every one of these calls used to be a bare, untimed
# subprocess.Popen, so a single stuck discovery handshake left that OS
# process running forever. `ps aux` on the reference VM turned up dozens of
# `gz-transport-topic`/`ros_gz_sim create` processes hours old, surviving
# every Ctrl-C restart in between - each one still holding a discovery
# registration, which made discovery for every NEW call slower too (a
# self-reinforcing pile-up, the same class of problem as bug #2 in
# docs/GAZEBO.md, just not fully closed off by cutting attempt counts
# alone). That pile-up is what actually starved the camera render loop and
# tanked RTF, which is what surfaces as "почти всё уходит в fail-safe" and
# "товары спавнятся и разлетаются" - not a classification or physics bug.
# `timeout -k` sends TERM at the deadline and escalates to KILL 1s later
# for anything that ignores it, so nothing from this node can wedge open
# past this ceiling regardless of how the launch process tree is (or
# isn't) reaped on shutdown.
GZ_CALL_TIMEOUT_S = 1.5
SPAWN_TIMEOUT_S = 15.0

PADDLE_TOPIC = {"shiber1_gabarit": "/shiber1/paddle_cmd", "shiber2_krug": "/shiber2/paddle_cmd"}
PADDLE_OPEN_ANGLE = {"shiber1_gabarit": -1.2, "shiber2_krug": 1.2}
PADDLE_VISIBLE_S = 1.2

# Divert: pin X at the shiber and slide into the cage via set_pose.
DIVERT_VY = {"shiber1_gabarit": 1.0, "shiber2_krug": -1.0}  # sign only
DIVERT_X = {"shiber1_gabarit": csim.X_SHIBER1, "shiber2_krug": csim.X_SHIBER2}
DIVERT_TARGET_Y = 1.0
DIVERT_SLIDE_STEPS = 10
DIVERT_SLIDE_DT = 0.08

# Serialize belt set_pose so GUI / discovery stay healthy.
POSE_DRIVE_DT = 0.20
ITEM_RIDE_Z = BELT_TOP_Z + 0.12
MAX_POSE_DRIVES_PER_TICK = 1
REAL_POSE_POLL_INTERVAL_S = 1.0
SPAWN_CLEAR_RADIUS_M = 0.35
STUCK_TIMEOUT_S = 20.0
STUCK_MAX_X = csim.X_SHIBER1


class GzBridgeNode(Node):
    def __init__(self):
        super().__init__("gz_bridge_node")
        self.declare_parameter("catalog_path", "")
        self.declare_parameter("pitch_m", csim.ITEM_PITCH_M)
        self.declare_parameter("belt_speed_mps", csim.BELT_SPEED_MPS)
        self.declare_parameter("seed", 7)
        self.declare_parameter("world", WORLD_NAME)
        # 0 = unlimited (default demo). Set to 1 for a single-item smoke test.
        self.declare_parameter("max_items", 0)
        # Optional exact catalog name (e.g. "Короб 300х200х200"). Empty =
        # pick randomly as before.
        self.declare_parameter("item_name", "")
        # Optional fixed yaw in degrees. NaN (default) = _safe_yaw() random.
        self.declare_parameter("yaw_deg", float("nan"))

        self.world = self.get_parameter("world").value
        self.pitch = float(self.get_parameter("pitch_m").value)
        self.speed = float(self.get_parameter("belt_speed_mps").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.max_items = int(self.get_parameter("max_items").value)
        self.item_name = str(self.get_parameter("item_name").value or "").strip()
        self.yaw_deg = float(self.get_parameter("yaw_deg").value)

        catalog_path = self.get_parameter("catalog_path").value or None
        self.catalog = sd.get_or_build_catalog(catalog_path)
        if self.item_name:
            match = [it for it in self.catalog if it.name == self.item_name]
            if not match:
                names = ", ".join(it.name for it in self.catalog)
                raise RuntimeError(
                    f"item_name={self.item_name!r} not in catalog; known: {names}")
            self.catalog = match

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
        # See _spawn()/_wait_and_kick(): only one `ros_gz_sim create` may be
        # in flight at a time.
        self._spawn_in_flight = False
        self.dt = 0.05
        self._pose_drive_cursor = 0
        self._pose_in_flight = False

        # See REAL_POSE_POLL_INTERVAL_S/SPAWN_CLEAR_RADIUS_M/STUCK_TIMEOUT_S
        # comment above.
        self.real_positions = {}     # gz_name -> (x, y, z), from the last poll
        self._progress_x = {}        # gz_name -> real x at its last MEANINGFUL move
        self._progress_t = {}        # gz_name -> wall-clock time.time() of that move
        # Drop any leftover item* models from a previous unclean shutdown
        # before we start spawning - otherwise the first spawn lands on
        # top of them (see _spawn_point_clear). Best-effort, non-blocking.
        threading.Thread(target=self._purge_stale_items, daemon=True).start()
        threading.Thread(target=self._pose_poll_loop, daemon=True).start()

        self.create_timer(self.dt, self._tick)
        self.create_timer(1.0, self._cleanup)
        scope = (f"max_items={self.max_items}" if self.max_items > 0 else "max_items=∞")
        fixed = f", fixed={self.item_name!r}" if self.item_name else ""
        self.get_logger().info(
            f"gz_bridge_node ready, {len(self.catalog)} item types, "
            f"world={self.world}, {scope}{fixed}, drive=kinematic, "
            f"pitch={self.pitch:.2f}m (~{self.pitch / self.speed:.1f}s @ {self.speed:.1f}m/s)")

    def _purge_stale_items(self):
        """Best-effort removal of any `item*` models already in the world
        at node start (leftovers from a previous run whose gz-sim process
        or /remove RPC didn't die cleanly)."""
        try:
            out = subprocess.run(
                ["gz", "model", "--list"],
                capture_output=True, text=True, timeout=5.0).stdout
        except Exception:
            return
        for line in out.splitlines():
            name = line.strip().lstrip("- ").strip()
            if not name.startswith("item"):
                continue
            self.get_logger().warn(f"purging leftover model {name!r} from world")
            subprocess.run(
                ["timeout", "-k", "1", "5", "gz", "service",
                 "-s", f"/world/{self.world}/remove",
                 "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "300", "--req", f'name: "{name}" type: MODEL'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # -- real-pose polling (spawn-point clearance + stuck-item detection) --
    PROGRESS_EPS_M = 0.03   # smaller moves than this don't count as "progressing" (jitter/settling)

    def _pose_poll_loop(self):
        # gz.msgs.Pose text format omits zero fields - a model sitting on
        # the belt centreline (y=0) is printed as `position { x: ... z: ... }`
        # with NO y line. The previous regex required y and silently dropped
        # every on-centreline item from real_positions - which, after we
        # stopped falling back to analytic x, meant triggers never fired
        # and items looked permanently stuck at spawn. y is optional now.
        pattern = re.compile(
            r'name:\s*"([^"]+)"\s*(?:id:\s*\d+\s*)?position\s*\{\s*'
            r'x:\s*(-?[\d.eE+-]+)\s*'
            r'(?:y:\s*(-?[\d.eE+-]+)\s*)?'
            r'z:\s*(-?[\d.eE+-]+)',
            re.DOTALL)
        while rclpy.ok():
            try:
                out = subprocess.run(
                    ["gz", "topic", "-e", "-t", f"/world/{self.world}/pose/info", "-n", "1"],
                    capture_output=True, text=True, timeout=3.0).stdout
                positions = {}
                for name, x, y, z in pattern.findall(out):
                    positions[name] = (float(x), float(y or 0.0), float(z))
                self.real_positions = positions
                tracked = [n for n in positions if n.startswith("item")]
                if tracked and not hasattr(self, "_pose_logged"):
                    self._pose_logged = True
                    self.get_logger().info(
                        f"pose poll seeing items: {tracked[:5]}")

                now = time.time()
                for name, (x, _y, _z) in positions.items():
                    it = self.items.get(name)
                    if it is None or it["routed_t"] is not None:
                        continue
                    if it.get("motion_start_t") is None:
                        continue
                    last_x = self._progress_x.get(name)
                    if last_x is None or abs(x - last_x) > self.PROGRESS_EPS_M:
                        self._progress_x[name] = x
                        self._progress_t[name] = now
                for name in list(self._progress_t):
                    if name not in self.items or self.items[name]["routed_t"] is not None:
                        self._progress_x.pop(name, None)
                        self._progress_t.pop(name, None)
            except Exception as exc:
                try:
                    self.get_logger().warn(
                        f"pose poll failed: {exc}", throttle_duration_sec=5.0)
                except Exception:
                    pass
            time.sleep(REAL_POSE_POLL_INTERVAL_S)

    def _spawn_point_clear(self):
        """True iff nothing physical is sitting on the spawn pose.

        IMPORTANT: check EVERY model in the real-pose poll whose name looks
        like one of ours (`itemN_...`), not just entries still tracked in
        `self.items`. Verified live: a leftover model from a previous run
        (or a force-despawn whose `/world/.../remove` RPC was lost) stays
        in gz-sim at x≈0.05, is invisible to the `name not in self.items`
        filter, and the next spawn then lands on top of it and gets stuck
        at x≈0.02 forever - which is exactly what the single-item smoke
        test hit (`item95_detergent` still in the world while item0 was
        being force-despawned as "no progress").
        """
        for name, (x, y, _z) in self.real_positions.items():
            if not name.startswith("item"):
                continue
            if abs(x - 0.05) < SPAWN_CLEAR_RADIUS_M and abs(y) < SPAWN_CLEAR_RADIUS_M:
                return False
        return True

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
        self._spawn_in_flight = True
        item_type = self.catalog[self.rng.integers(0, len(self.catalog))]
        slug = self.slug_by_name.get(item_type.name)
        if slug is None:
            self.get_logger().warn(
                f"no gz model slug for catalog item {item_type.name!r} - skipping")
            self._spawn_in_flight = False
            return
        gz_name = f"item{self.next_id}_{slug}"
        self.next_id += 1
        if self.yaw_deg == self.yaw_deg:   # not NaN
            yaw = float(np.radians(self.yaw_deg))
        else:
            yaw = self._safe_yaw(item_type.dims_mm)
        shortest_m = min(item_type.dims_mm) / 1000.0
        spawn_z = BELT_TOP_Z + 0.025 + shortest_m / 2.0 + SPAWN_CLEARANCE_M
        model_file = os.path.join(self.models_dir, slug, "model.sdf")

        cmd = ["timeout", "-k", "1", f"{SPAWN_TIMEOUT_S:.0f}",
               "ros2", "run", "ros_gz_sim", "create",
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
            "yaw": yaw, "x": 0.0, "y": 0.0, "sensed": False, "spawn_t": self.t,
            "motion_start_t": None, "routed_t": None,
            "ride_z": spawn_z, "last_pose_drive_t": -1e9,
        }
        self.get_logger().info(f"spawn {gz_name} ({item_type.name}, yaw={np.degrees(yaw):.0f} deg)")

        threading.Thread(target=self._wait_and_kick, args=(gz_name, proc), daemon=True).start()

    def _wait_and_kick(self, gz_name, proc):
        proc.wait()
        self._spawn_in_flight = False
        it = self.items.get(gz_name)
        if it is None:
            return
        if proc.returncode != 0:
            self.get_logger().warn(
                f"{gz_name}: spawn ('ros_gz_sim create') failed or timed out "
                f"(exit {proc.returncode}) - dropping")
            it["spawn_failed"] = True
            return
        it["motion_start_t"] = self.t
        self.get_logger().info(f"motion start {gz_name} (kinematic)")

    def _set_pose(self, gz_name, x, y, z, yaw, wait=False):
        """Teleport a model. Async calls are serialized via _pose_in_flight."""
        if not wait:
            if self._pose_in_flight:
                return False
            self._pose_in_flight = True
        half = yaw * 0.5
        qw, qz = float(np.cos(half)), float(np.sin(half))
        req = (f'name: "{gz_name}", '
               f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
               f'orientation: {{x: 0, y: 0, z: {qz:.6f}, w: {qw:.6f}}}')

        def _run():
            try:
                subprocess.run(
                    ["timeout", "-k", "1", "1", "gz", "service",
                     "-s", f"/world/{self.world}/set_pose",
                     "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                     "--timeout", "200", "--req", req],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=2.0)
            except Exception:
                pass
            finally:
                if not wait:
                    self._pose_in_flight = False

        if wait:
            _run()
            return True
        threading.Thread(target=_run, daemon=True).start()
        return True

    def _push(self, gz_name, shiber):
        """Slide the item into its roll cage at the shiber X."""
        sign = 1.0 if DIVERT_VY[shiber] > 0 else -1.0
        target_y = sign * DIVERT_TARGET_Y
        shiber_x = DIVERT_X[shiber]
        it = self.items.get(gz_name)
        yaw = it["yaw"] if it else 0.0
        z0 = it.get("ride_z", ITEM_RIDE_Z) if it else ITEM_RIDE_Z

        def run():
            y0 = 0.0
            real = self.real_positions.get(gz_name)
            if real is not None:
                y0 = real[1]
            for i in range(1, DIVERT_SLIDE_STEPS + 1):
                y = y0 + (target_y - y0) * (i / DIVERT_SLIDE_STEPS)
                self._set_pose(gz_name, shiber_x, y, z0, yaw, wait=True)
                if it is not None:
                    it["x"], it["y"] = shiber_x, y
                time.sleep(DIVERT_SLIDE_DT)
            self.get_logger().info(
                f"divert done {gz_name} -> y={target_y:.1f} at x={shiber_x}")

        threading.Thread(target=run, daemon=True).start()

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
        self.items[gz_name]["divert_shiber"] = shiber
        self.get_logger().info(f"divert {gz_name} via {shiber}")
        self._push(gz_name, shiber)
        self._set_paddle(shiber, PADDLE_OPEN_ANGLE[shiber])

        def close_paddle():
            self._set_paddle(shiber, 0.0)
            paddle_timer.cancel()

        paddle_timer = self.create_timer(PADDLE_VISIBLE_S, close_paddle)

    def _name_for_item(self, item_id):
        for name, it in self.items.items():
            if it["item_id"] == item_id:
                return name
        return None

    def _gz_pub_burst(self, topic, msgtype, payload, attempts, interval):
        script = (f"for i in $(seq 1 {max(1, attempts)}); do "
                  f"timeout -k 1 {GZ_CALL_TIMEOUT_S} gz topic -t {topic} -m {msgtype} "
                  f"-p '{payload}'; "
                  f"{'sleep '+str(interval)+';' if interval > 0 else ''} done")
        subprocess.Popen(["bash", "-c", script],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _set_paddle(self, shiber, angle):
        topic = PADDLE_TOPIC.get(shiber)
        if topic is None:
            return
        self._gz_pub_burst(topic, "gz.msgs.Double", f"data: {angle}",
                            PADDLE_ATTEMPTS, PADDLE_INTERVAL_S)

    # -- housekeeping ------------------------------------------------------
    def _tick(self):
        self.t += self.dt
        if self.t >= self.next_spawn_t:
            reached_cap = self.max_items > 0 and self.next_id >= self.max_items
            if reached_cap:
                pass
            elif self._spawn_in_flight:
                pass
            elif self._spawn_point_clear():
                self._spawn()
                self.next_spawn_t += self.pitch / self.speed

        moving = []
        for gz_name, it in list(self.items.items()):
            if it["motion_start_t"] is not None and not it.get("spawn_failed"):
                if it["routed_t"] is None:
                    # On-belt: open-loop along centreline (y=0).
                    it["x"] = self.speed * (self.t - it["motion_start_t"])
                    it["y"] = 0.0
                    it["seen_real"] = True
                    moving.append(gz_name)
            else:
                real = self.real_positions.get(gz_name)
                if real is not None:
                    it["x"], it["y"] = real[0], real[1]
                    it["seen_real"] = True

            if (not it["sensed"]
                    and it.get("seen_real")
                    and it["x"] >= csim.X_SENSOR):
                it["sensed"] = True
                self.pub_sensor.publish(String(data=json.dumps({
                    "item_id": it["item_id"],
                    "t": self.t,
                    "type_name": it["type_name"],
                    "yaw": it["yaw"],
                }, ensure_ascii=False)))
                self.get_logger().info(
                    f"trigger {gz_name} at x={it['x']:.2f}")

        if moving and not self._pose_in_flight:
            n = len(moving)
            for _ in range(min(MAX_POSE_DRIVES_PER_TICK, n)):
                self._pose_drive_cursor = self._pose_drive_cursor % n
                gz_name = moving[self._pose_drive_cursor]
                self._pose_drive_cursor += 1
                it = self.items.get(gz_name)
                if it is None:
                    continue
                if (self.t - it["last_pose_drive_t"]) < POSE_DRIVE_DT:
                    continue
                if self._set_pose(gz_name, it["x"], 0.0,
                                   it.get("ride_z", ITEM_RIDE_Z), it["yaw"]):
                    it["last_pose_drive_t"] = self.t
                    break

        self.pub_position.publish(String(data=json.dumps({
            "t": round(self.t, 3),
            "items": [{"id": it["item_id"], "x": round(it["x"], 3)}
                      for it in self.items.values()],
        })))

    def _cleanup(self):
        to_remove = []
        for name, it in self.items.items():
            spawn_failed = it.get("spawn_failed", False)
            past_end = it["x"] > DESPAWN_X
            routed_stale = it["routed_t"] is not None and (self.t - it["routed_t"]) > CLEANUP_AFTER_S
            if spawn_failed or past_end or routed_stale:
                to_remove.append(name)
        for name in to_remove:
            it = self.items[name]
            self._progress_x.pop(name, None)
            self._progress_t.pop(name, None)
            if not it.get("spawn_failed", False):
                subprocess.Popen(
                    ["timeout", "-k", "1", "5", "gz", "service",
                     "-s", f"/world/{self.world}/remove",
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
