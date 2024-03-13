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
import numpy as np
from enum import Enum

class State(Enum):
    STOP = 0
    FOLLOW = 1


class Mover:
    def __init__(self):
        rospy.init_node("mover")
        # Raw data
        self.twist = Twist()

        self.command = 0
        self.imu = Imu()
        self.bumper = BumperEvent()
        self.cliff = CliffEvent()
        self.odom = Odometry()
        self.target = Twist()

        # Processed data
        self.yaw = None

        self.state = State.STOP

        # Data for movement
        self.rotate_p = 0.5
        self.linear_p = 1
        self.timer_target = 0.0
        self.timer = 0.0

        # Subscriber
        self.bumper_sub = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bumper_cb)
        self.cliff_sub = rospy.Subscriber('/mobile_base/events/cliff', CliffEvent, self.cliff_cb)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.imu_sub = rospy.Subscriber("/mobile_base/sensors/imu_data", Imu, self.imu_cb)
        self.command_sub = rospy.Subscriber("/internal_command", Int32, self.command_cb)
        self.target_sub = rospy.Subscriber("/target", Twist, self.target_cb)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=1)
    
    def wrap_to_pi(self, x):
        return np.mod(x+np.pi,2*np.pi)-np.pi

    def imu_cb(self, data):
        self.imu = data

    def target_cb(self, data):
        if self.state == State.FOLLOW:
            self.target = data.linear.x

    def odom_cb(self, data):
        self.odom = data

        quat = data.pose.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        self.yaw = yaw
      
    def bumper_cb(self, data):
        self.bumper = data
        if data.state == CliffEvent.CLIFF:
            self.ok = False
            # print "Cliff event: %s,%s"%(str(data.sensor),str(data.state))
            if   data.sensor == CliffEvent.LEFT:
                self.theta_goal = self.theta - 3.141592*random.uniform(0.2, 1.0)
            elif data.sensor == CliffEvent.RIGHT:
                self.theta_goal = self.theta + 3.141592*random.uniform(0.2, 1.0)
            else:
                self.theta_goal = self.wrap_to_pi(self.theta + 3.141592*random.uniform(-1.0, 1.0))

    def cliff_cb(self, data):
        self.cliff = data
        if data.state == BumperEvent.PRESSED:
            self.ok = False
            if data.bumper == BumperEvent.LEFT:
                self.theta_goal = 0
            elif data.bumper == BumperEvent.RIGHT:
                self.theta_goal = self.theta + 3.141592*random.uniform(0.2, 1.0)
            else:
                self.theta_goal = self.wrap_to_pi(self.theta + 3.141592*random.uniform(-1.0, 1.0))

    def command_cb(self, data):
        self.command = data.data
        rospy.loginfo(self.command)
        # change the states depends on current sate & command
        # if self.command.data == 0:
        #     self.state = State.STOP
        # elif self.command.data == 1:
        #     self.state = State.FOLLOW
        # elif self.command.data >= 2:
        #     self.state = State.ORDERED

    def move(self, vel):
        twist = Twist()
        twist.linear.x = vel
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)

    # make a function that follows the angle for s seconds. For s seconds, robot became Oredered mode

    def follow(self):

        speed = 0.1 # linear speed
        p = 0.5 # p gain for rotate

        if x is None:
            speed = 0
            x = 0

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = x * p
        self.cmd_vel_pub.publish(twist)

    def get_timer(self, set=None):
        if set is not None:
            self.timer_target = set
        if self.timer_target <= self.timer:
            self.timer_target = 0
            self.timer = 0
            return True
        return False
    
    def update_timer(self):
        if self.timer_target != 0:
            self.timer += 1 / 30

    def run(self):
        rate = rospy.Rate(30)
        # implement FSM
        while not rospy.is_shutdown():
            self.follow()
            self.update_timer()
            rate.sleep()


if __name__ == "__main__":
    node = Mover()
    node.run()
    rospy.spin()