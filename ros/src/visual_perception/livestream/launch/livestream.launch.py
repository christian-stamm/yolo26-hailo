import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch StreamDemo C++ node using a YAML parameter file."""
    package_share = get_package_share_directory("livestream")
    default_params = os.path.join(package_share, "config", "livestream.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to YAML file with stream demo parameters",
    )

    stream_demo_node = Node(
        package="livestream",
        executable="livestream_node",
        name="StreamDemo",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription(
        [
            params_file_arg,
            stream_demo_node,
        ]
    )
