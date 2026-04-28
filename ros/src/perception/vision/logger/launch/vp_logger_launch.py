from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

PACKAGE_GROUP = "vp"
PACKAGE_NAME = "boxlogger"

def generate_launch_description():
    package_share = get_package_share_directory(f"{PACKAGE_GROUP}_{PACKAGE_NAME}")
    default_params = os.path.join(package_share, "config", f"{PACKAGE_NAME}_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to YAML file with node parameters",
    )

    node = Node(
        package=f"{PACKAGE_GROUP}_{PACKAGE_NAME}",
        executable=f"{PACKAGE_NAME}_node",
        name=f"{PACKAGE_NAME}_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([
        params_file_arg,
        node,
    ])
