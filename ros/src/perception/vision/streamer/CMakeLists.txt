cmake_minimum_required(VERSION 3.8)
project(vp_streamer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(message_filters REQUIRED)
find_package(vp_interface REQUIRED)

include_directories(include)

# Stream Demo Node - optimized with zero-copy and intra-process comms
add_executable(${PROJECT_NAME}_node
  src/streamer_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  ${OpenCV_LIBS}
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
  vp_interface
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
