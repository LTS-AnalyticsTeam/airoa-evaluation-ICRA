cmake_minimum_required(VERSION 3.0.2)
project(hsr_policy_client)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  rospy
  sensor_msgs
)

add_service_files(
  DIRECTORY srv
  FILES
    StringTrigger.srv
)

generate_messages()

catkin_package(
  CATKIN_DEPENDS
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
