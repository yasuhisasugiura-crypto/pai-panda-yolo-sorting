cmake_minimum_required(VERSION 3.8)
project(mtc_tutorial)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_task_constructor_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

add_executable(mtc_node src/mtc_node.cpp)
target_compile_features(mtc_node PUBLIC cxx_std_17)
target_link_libraries(mtc_node Eigen3::Eigen Threads::Threads)
ament_target_dependencies(mtc_node
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
  rclcpp
  rclcpp_action
  shape_msgs
  std_msgs
  visualization_msgs
  vision_msgs
)

install(TARGETS mtc_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch worlds models config DESTINATION share/${PROJECT_NAME})

ament_package()

