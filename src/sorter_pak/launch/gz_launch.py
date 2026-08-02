"""Full Gazebo (gz-sim / Harmonic) run of the PAK line: real rigid-body
physics for the belt + items + shiber pushes, driving the *same*
perception_node/executive_node used by the pure-Python simulation.

Prerequisites (see docs/GAZEBO.md):
  - ROS2 Jazzy + Gazebo Harmonic + ros_gz (ros_gz_sim, ros_gz_bridge)
  - `python3 tools/export_gz_models.py` already run at least once
  - package built with `colcon build --symlink-install`

Usage:
    ros2 launch sorter_pak gz_launch.py
    ros2 launch sorter_pak gz_launch.py gui:=false     # headless (server-only)
    # single-item smoke test (known B-category box):
    ros2 launch sorter_pak gz_launch.py gui:=false max_items:=1 item_name:='Короб 300х200х200'

Then, in another shell:
    ros2 topic echo /pak/routing_result
"""
import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# gz-sim needs a few seconds to load the world and bring up SceneBroadcaster
# (which is what actually publishes /world/.../dynamic_pose/info and the 2
# camera topics) before anything can usefully bridge/subscribe to it. All
# ROS2-side nodes below are started via TimerAction after this delay rather
# than at t=0 - starting a ros_gz_bridge parameter_bridge before the gz-side
# topic exists was empirically observed to leave it silently never
# receiving data for the rest of the run (bridge advertises fine on the
# ROS2 side, but never actually gets the gz-side subscription to stick).
GZ_STARTUP_DELAY_S = 5.0


def _cleanup_stale_processes(world_path):
    """Verified live on the reference VM: `gz sim` (the actual simulation
    server, a descendant of the `ruby .../gz sim ...` process this launch
    file starts) does not reliably die when `ros2 launch` is stopped -
    neither Ctrl-C (SIGINT) nor `timeout`'s SIGTERM reliably reach/kill it;
    it was observed still running and consuming ~50% CPU minutes after a
    bounded test launch had exited. A leftover `gz sim` instance loads the
    same world and advertises the exact same gz-transport topic/service
    names (world, /model/<name>/cmd_vel, pose/info, ...) as a freshly
    started one - this project's spawn/kick/pose-poll calls then land on
    WHICHEVER instance answers first, non-deterministically, which is a
    very plausible root cause of "items spawn and immediately scatter"
    (two independent physics worlds fighting over what look like the same
    model names) as well as of silently-lost kicks. Run once, synchronously,
    before any launch actions start, so every `ros2 launch` begins from a
    guaranteed-clean slate regardless of how the previous run ended.

    Patterns match ANY `gz sim ...pak_line.sdf` (not just this install's
    absolute path) because the same world has historically been launched
    from both `/mnt/share/install/...` and `~/ros2_ws/install/...` on the
    reference VM - a path-scoped pkill left the other copy alive and its
    leftover `itemN_*` models then blocked the spawn point of the fresh
    run (see gz_bridge_node._spawn_point_clear).
    """
    patterns = [
        "gz sim .*pak_line\\.sdf",
        "lib/sorter_pak/gz_bridge_node",
        "lib/sorter_pak/perception_node",
        "lib/sorter_pak/executive_node",
    ]
    for pattern in patterns:
        subprocess.run(["pkill", "-9", "-f", pattern],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Also reap orphaned one-shot CLI helpers left by a previous bridge
    # (same self-congestion class of bug - see GZ_CALL_TIMEOUT_S comment
    # in gz_bridge_node.py).
    for pattern in ("gz-transport-topic", "ros_gz_sim create"):
        subprocess.run(["pkill", "-9", "-f", pattern],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_launch_description():
    pkg_share = get_package_share_directory("sorter_pak")
    world_path = os.path.join(pkg_share, "worlds", "pak_line.sdf")
    _cleanup_stale_processes(world_path)

    gui_arg = DeclareLaunchArgument("gui", default_value="true")
    seed_arg = DeclareLaunchArgument("seed", default_value="7")
    max_items_arg = DeclareLaunchArgument(
        "max_items", default_value="0",
        description="0=unlimited; set 1 for a single-item smoke test")
    item_name_arg = DeclareLaunchArgument(
        "item_name", default_value="",
        description="Optional exact catalog name; empty = random")
    yaw_deg_arg = DeclareLaunchArgument(
        "yaw_deg", default_value="nan",
        description="Fixed spawn yaw in degrees; nan = safe random")
    classification_source_arg = DeclareLaunchArgument(
        "classification_source", default_value="auto",
        description="vision | synthetic | auto (vision then synthetic fallback)")
    pitch_m_arg = DeclareLaunchArgument(
        "pitch_m", default_value="3.0",
        description="Metres between spawns (demo-friendly). TZ nominal ~0.7; "
                    "use pitch_m:=0.7 for full-rate. At 1 m/s, 3.0 m ≈ 3 s gap.")
    gui = LaunchConfiguration("gui")
    seed = LaunchConfiguration("seed")
    max_items = LaunchConfiguration("max_items")
    item_name = LaunchConfiguration("item_name")
    yaw_deg = LaunchConfiguration("yaw_deg")
    classification_source = LaunchConfiguration("classification_source")
    pitch_m = LaunchConfiguration("pitch_m")

    # ros_gz_sim's gz_args argument-passing differs slightly across
    # releases, so gui vs headless are two explicit Includes (each only
    # active via its own condition) rather than one conditionally-built
    # argument string.
    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([get_package_share_directory("ros_gz_sim"),
                                   "launch", "gz_sim.launch.py"])),
        launch_arguments={"gz_args": [world_path, " -r"]}.items(),
        condition=IfCondition(gui),
    )
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([get_package_share_directory("ros_gz_sim"),
                                   "launch", "gz_sim.launch.py"])),
        launch_arguments={"gz_args": [world_path, " -r -s"]}.items(),
        condition=UnlessCondition(gui),
    )

    # NOTE: no gz.msgs.Pose_V -> tf2_msgs/msg/TFMessage bridge here on
    # purpose. It was tried first, but on the reference VM that conversion
    # drops the per-pose `name` field (both header.frame_id and
    # child_frame_id came back empty), so there was no reliable way to
    # match a transform back to a spawned item from the ROS2 side. See
    # gz_bridge_node.py's module docstring ("Why time-based position, not
    # the pose bridge") for the analytic-position approach used instead.

    # The 2 real RGB cameras (worlds/pak_line.sdf: camera_top/camera_side)
    # -> sensor_msgs/msg/Image (rgb8), consumed directly by perception_node.py
    # (sorter_pak/core/vision.py) - this is the actual CV pipeline, not a
    # ground-truth shortcut. Segmentation/measurement is the same
    # appearance+contour approach as https://github.com/irwkc/robozone
    # (a working real-camera classifier for this exact track), adapted for
    # a simulated camera at a known fixed pose (see core/vision.py). Topic
    # strings here MUST match the <topic> set on each <sensor> in
    # worlds/pak_line.sdf exactly (no leading slash, so ROS2's default
    # namespace resolution gives the same /camera_.../... name on both the
    # perception_node subscription and this bridge).
    camera_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="pak_camera_bridge",
        output="screen",
        arguments=[
            "camera_top/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "camera_side/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "camera_overview/image@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
    )

    # CRITICAL: without this, every node below runs on wall-clock time
    # while gz-sim (physics + software-rendered cameras on a machine with
    # no GPU) can fall far behind real time - on the reference VM real
    # time factor was measured at ~2.5% (i.e. simulated motion is ~40x
    # slower than the wall clock). gz_bridge_node's analytic position
    # estimate (x = speed*(t-motion_start_t)) and perception_node's
    # "is this camera frame fresh" check both used to read wall-clock
    # time, so they raced ~40x ahead of what had actually physically
    # happened - items appeared to "pile up" at the spawn point (the
    # *belief* about where they were kept advancing on schedule even
    # though gz-sim had barely moved them), and nearly every
    # classification fell through to the fail-safe (stale/missing camera
    # frame relative to the wall-clock-based trigger time). Bridging
    # /clock and setting use_sim_time=true on every node below makes
    # rclpy's timers and self.get_clock().now() follow gz-sim's actual
    # simulated clock instead, however slow that turns out to run.
    clock_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="pak_clock_bridge",
        output="screen",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )

    sim_time = {"use_sim_time": True}
    bridge_node = Node(
        package="sorter_pak", executable="gz_bridge_node", name="gz_bridge_node",
        output="screen",
        parameters=[{
            "seed": seed,
            "max_items": ParameterValue(max_items, value_type=int),
            "item_name": ParameterValue(item_name, value_type=str),
            "yaw_deg": ParameterValue(yaw_deg, value_type=float),
            "pitch_m": ParameterValue(pitch_m, value_type=float),
            **sim_time,
        }],
    )
    perception_node = Node(
        package="sorter_pak", executable="perception_node", name="perception_node",
        output="screen",
        parameters=[{
            "classification_source": ParameterValue(classification_source, value_type=str),
            **sim_time,
        }],
    )
    executive_node = Node(
        package="sorter_pak", executable="executive_node", name="executive_node",
        output="screen", parameters=[sim_time],
    )

    delayed_ros_side = TimerAction(
        period=GZ_STARTUP_DELAY_S,
        actions=[clock_bridge, camera_bridge, bridge_node, perception_node, executive_node],
    )

    return LaunchDescription([
        gui_arg, seed_arg, max_items_arg, item_name_arg, yaw_deg_arg,
        classification_source_arg, pitch_m_arg,
        gz_sim_gui, gz_sim_headless,
        delayed_ros_side,
    ])
