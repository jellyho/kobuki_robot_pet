cmake_minimum_required(VERSION 2.8.3)
project(robot_pet)
find_package(catkin REQUIRED COMPONENTS kobuki_driver kobuki_node std_msgs sensor_msgs geometry_msgs message_generation)

catkin_python_setup()
catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS scripts/mover.py
                 scripts/perceiver_keyop.py
                 scripts/perceiver.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
