cmake_minimum_required(VERSION 3.8)
project(beni_head CXX)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(head SHARED src/head.cpp)
ament_target_dependencies(head rclcpp rclcpp_components geometry_msgs sensor_msgs)
rclcpp_components_register_node(head PLUGIN "beni::Head" EXECUTABLE head_node)

install(TARGETS head ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_package()
