"""Brings up the full simulated PAK: conveyor_sim_node (plant) +
perception_node (CV/classification) + executive_node (2-shiber control).

Usage (after `colcon build --symlink-install && source install/setup.bash`):

    ros2 launch sorter_pak sim_launch.py
    ros2 launch sorter_pak sim_launch.py seed:=42 catalog_path:=/path/to/item_catalog.json

Then, in another shell, e.g.:

    ros2 topic echo /pak/routing_result
    ros2 topic hz /pak/item/position
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    catalog_path_arg = DeclareLaunchArgument("catalog_path", default_value="")
    seed_arg = DeclareLaunchArgument("seed", default_value="7")

    catalog_path = LaunchConfiguration("catalog_path")
    seed = LaunchConfiguration("seed")

    return LaunchDescription([
        catalog_path_arg,
        seed_arg,
        Node(
            package="sorter_pak", executable="conveyor_sim_node", name="conveyor_sim_node",
            output="screen",
            parameters=[{"catalog_path": catalog_path, "seed": seed}],
        ),
        Node(
            package="sorter_pak", executable="perception_node", name="perception_node",
            output="screen",
            parameters=[{"catalog_path": catalog_path}],
        ),
        Node(
            package="sorter_pak", executable="executive_node", name="executive_node",
            output="screen",
        ),
    ])
