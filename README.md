# 24-1 Graduation Project

ros noetic(ubuntu 20.04)


### 1. Installation
```
mkdir catkin_ws
cd catkin_ws
mkdir src
cd src
git clone --recursive https://github.com/jellyho/kobuki_robot_pet.git
sudo apt get install ros-noetic-ecl-core
sudo apt install liborocos-kdl-dev
sudo apt install ros-noetic-joy
rosdep install --from-paths . --ignore-src -r -y
cd ..
catkin_make

rosrun kobuki_ftdi create_udev_rules
```

```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

### 2. Run

> Before run this code, make sure you connected kobuki via usb and turned on!

```
roslaunch robot_pet wakeup.launch
```
