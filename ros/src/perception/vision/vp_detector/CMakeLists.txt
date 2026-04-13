cmake_minimum_required(VERSION 3.16)
project(vp_detector)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FindPkgConfig)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio imgproc)

find_package(ament_cmake REQUIRED)
find_package(vp_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(yolo26_detector STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detector_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hailo/backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vision/preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vision/postprocess.cpp
)

add_library(yolo26::yolo26 ALIAS yolo26_detector)

target_include_directories(yolo26_detector
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc

    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo26_detector
    PUBLIC
    HailoRT::libhailort
    ${OpenCV_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node
  ${CMAKE_CURRENT_SOURCE_DIR}/src/detector_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  yolo26::yolo26
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  cv_bridge
  vp_interface
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
