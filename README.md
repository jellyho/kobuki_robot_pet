# 24-1 Graduation Project

ros noetic(ubuntu 20.04)


### 1. Installation
```
mkdir catkin_ws
cd catkin_ws
mkdir src
git clone https://github.com/jellyho/kobuki_robot_pet.git
cd ..
catkin_make
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