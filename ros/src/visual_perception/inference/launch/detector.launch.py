from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share = get_package_share_directory("visual_perception_inference")
    default_params = os.path.join(package_share, "config", "detector_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to YAML file with node parameters",
    )

    node = Node(
        package="visual_perception_inference",
        executable="yolo26_detector_node",
        name="yolo26_detector_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([
        params_file_arg,
        node,
    ])
