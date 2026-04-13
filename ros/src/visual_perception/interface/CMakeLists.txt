cmake_minimum_required(VERSION 3.16)
project(visual_perception_interface)

include(FindPkgConfig)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BBoxDetList.msg"
  "msg/BBoxDet.msg"
  "srv/BBoxDet.srv"
  DEPENDENCIES std_msgs sensor_msgs vision_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

ament_package()