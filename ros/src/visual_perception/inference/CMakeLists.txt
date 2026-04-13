cmake_minimum_required(VERSION 3.16)
project(visual_perception_inference)

include(FindPkgConfig)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio imgproc)

find_package(ament_cmake REQUIRED)
find_package(visual_perception_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(yolo26_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detector_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hailo/backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vision/preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vision/postprocess.cpp
)

add_library(yolo26::yolo26 ALIAS yolo26_core)

target_include_directories(yolo26_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc

    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

  target_link_libraries(yolo26_core
    PUBLIC
    HailoRT::libhailort
    ${OpenCV_LIBRARIES}
)

add_executable(yolo26_detector_node
  ${CMAKE_CURRENT_SOURCE_DIR}/src/detector_node.cpp
)

target_link_libraries(yolo26_detector_node
  yolo26::yolo26
)

ament_target_dependencies(yolo26_detector_node
  rclcpp
  cv_bridge
  visual_perception_interface
)

install(TARGETS yolo26_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
