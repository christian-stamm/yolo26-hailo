cmake_minimum_required(VERSION 3.16)
project(vp_evaluation)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

find_package(ament_cmake REQUIRED)
find_package(vp_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(${PROJECT_NAME}_node
  src/evaluation_node.cpp
)

target_include_directories(${PROJECT_NAME}_node
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(${PROJECT_NAME}_node
  nlohmann_json::nlohmann_json
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  vp_interface
  sensor_msgs
  vision_msgs
  std_msgs
  cv_bridge
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
