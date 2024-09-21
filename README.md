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

sudo apt-get install ros-noetic-realsense2-camera
sudo apt-get install ros-noetic-realsense2-description

cd src
git clone --recursive https://github.com/jellyho/yolov8-ros1.git
pip install ultralytics
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

![image](https://github.com/user-attachments/assets/048ea613-027f-489a-851f-9866c9350e70)

![image](https://github.com/user-attachments/assets/c59cb6ae-fa21-4965-80fa-74df451a0c86)
![image](https://github.com/user-attachments/assets/4df57d98-5352-47e3-a268-7061a6b50dc4)






