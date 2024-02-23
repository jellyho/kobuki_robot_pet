#!/usr/bin/env python3

import roslib; roslib.load_manifest('kobuki_testsuite')
import rospy, sys, random

from tf.transformations import euler_from_quaternion
from math import degrees

from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from kobuki_msgs.msg import BumperEvent, CliffEvent
from enum import Enum

class State(Enum):
    STOP = 0
    SEARCH = 1
    ORDERED = 2
    FOLLOW = 3


class Mover:
    def __init__(self):
        rospy.init_node("mover")

        # Raw data
        self.twist = Twist()

        self.command = Int32()
        self.imu = Imu()
        self.bumper = BumperEvent()
        self.cliff = CliffEvent()
        self.odom = Odometry()

        # Processed data
        self.yaw = None

        self.state = State.STOP

        # Data for movement
        self.distance_counter = 0
        self.distance_target = 0

        # Subscriber
        self.bumper_sub = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bumper_cb)
        self.cliff_sub = rospy.Subscriber('/mobile_base/events/cliff', CliffEvent, self.cliff_cb)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.imu_sub = rospy.Subscriber("/mobile_base/sensors/imu_data", Imu, self.imu_cb)
        self.command_sub = rospy.Subscriber("/internal_command", Int32, self.command_cb)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=1)
    
    def imu_cb(self, data):
        self.imu = data

    def odom_cb(self, data):
        self.odom = data

        quat = data.pose.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        self.theta = yaw
      
    def bumper_cb(self, data):
        self.bumper = data

    def cliff_cb(self, data):
        self.cliff = data

    def command_cb(self, data):
        self.command = data
        rospy.loginfo(self.command.data)
        if self.command.data == 0:
            self.state = State.STOP
        elif self.command.data >= 1:
            self.state = State.ORDERED

    def move(self, vel):
        twist = Twist()
        twist.linear.x = vel
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.state == State.ORDERED:
                if self.command.data == 1:
                    self.move(0.3)
            else:
                self.move(0)
            rate.sleep()


if __name__ == "__main__":
    node = Mover()
    node.run()
    rospy.spin()