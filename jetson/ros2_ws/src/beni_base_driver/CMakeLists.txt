cmake_minimum_required(VERSION 3.8)
project(beni_base_driver CXX)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# shared/proto/base_proto.h: the repo tree when built in place, /ws/shared in the container (-DBENI_SHARED_DIR).
set(BENI_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared" CACHE PATH "repo shared/ directory")
if(NOT EXISTS "${BENI_SHARED_DIR}/proto/base_proto.h")
  message(FATAL_ERROR "base_proto.h not found under BENI_SHARED_DIR=${BENI_SHARED_DIR}")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_library(base_driver SHARED src/base_driver.cpp)
target_include_directories(base_driver PRIVATE "${BENI_SHARED_DIR}/proto")
ament_target_dependencies(base_driver rclcpp rclcpp_components geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs)
rclcpp_components_register_node(base_driver PLUGIN "beni::BaseDriver" EXECUTABLE base_driver_node)

install(TARGETS base_driver ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_package()
