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
    ORDERED = 2
    WAIT = 3
    EMERGENCY = 4


class Mover:
    def __init__(self):
        rospy.init_node("mover")
        # Raw data
        self.command = 0
        self.imu = Imu()
        self.bumper = BumperEvent()
        self.cliff = CliffEvent()
        self.odom = Odometry()

        self.current = Twist() # linear.x(velocity target), angular.z(angle target)
        self.current.linear.x = 0
        self.current.angular.z = 0
        self.target = Twist()
        self.target.linear.x = 0
        self.target.angular.z = 0

        # Processed data
        self.yaw = None

        self.state = State.STOP

        # Data for movement
        self.rotate_p = 0.5
        self.linear_p = 0.3
        self.wandering_offset = 0.5 # 0.5pi = 90degrees
        self.timer_target = 0.0
        self.timer = 0.0

        # Subscriber
        self.bumper_sub = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bumper_cb)
        self.cliff_sub = rospy.Subscriber('/mobile_base/events/cliff', CliffEvent, self.cliff_cb)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.command_sub = rospy.Subscriber("/internal_command", Int32, self.command_cb)
        self.target_sub = rospy.Subscriber("/target", Twist, self.target_cb)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=1)
    
    def wrap_to_pi(self, x):
        return np.mod(x+np.pi,2*np.pi)-np.pi
    
    def wrap_to_2pi(x):
        return np.mod(x + 2*np.pi, 4*np.pi) - 2*np.pi

    def target_cb(self, data): # following target
        if self.state == State.FOLLOW:
            self.target.angular.z = self.target.angular.z + data.angular.z # update target

    def odom_cb(self, data):
        self.odom = data
        quat = data.pose.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        self.current.angular.z = yaw
      
    def bumper_cb(self, data):
        self.bumper = data
        if data.state == CliffEvent.CLIFF:
            self.state = State.EMERGENCY
            print (f"Cliff event: {str(data.sensor)},{str(data.state)}")
            if data.sensor == CliffEvent.LEFT:
                self.target.angular.z = self.current.angular.z + 3.141592 * self.wandering_offset
            elif data.sensor == CliffEvent.RIGHT:
                self.target.angular.z = self.current.angular.z - 3.141592 * self.wandering_offset
            else:
                self.target.angular.z = self.current.angular.z - 3.141592

    def cliff_cb(self, data):
        self.cliff = data
        if data.state == BumperEvent.PRESSED:
            self.state = State.EMERGENCY
            if data.bumper == BumperEvent.LEFT:
                self.target.angular.z = self.current.angular.z + 3.141592 * self.wandering_offset
            elif data.bumper == BumperEvent.RIGHT:
                self.target.angular.z = self.current.angular.z - 3.141592 * self.wandering_offset
            else:
                self.target.angular.z = self.current.angular.z - 3.141592

    def set_target(self, x=None, r=None):
        self.target.linear.x = x if x is not None else self.target.linear.x
        self.target.angular.z = r if r is not None else self.target.angular.z

    def action_spin(self):
        self.set_target(0, self.current.angular.z + 3.141592)
        self.set_timer(2)

    def command_cb(self, data):
        self.command = data.data
        # implement FSM
        if self.state == State.STOP: #STOP
            if self.command.data == 0:
                self.state = State.STOP
            elif self.command.data == 1:
                self.state = State.FOLLOW
            elif self.command.data >= 2:
                if self.command.data == 2:
                    self.action_spin()
                self.state = State.WAIT
        elif self.state == State.EMERGENCY: #EMERGENCY
            self.set_target(x = -0.5)
            self.set_timer(1)
            self.state == State.WAIT
        elif self.state == State.WAIT: #WAIT
            if self.check_timer():
                self.state == State.STOP  
        elif self.state == State.FOLLOW:
            if self.command.data != 1:
                self.state = State.STOP

    def follow(self):
        # load current
        c_x = self.current.linear.x # curretn velocity
        t_x = self.target.linear.x # target velocity

        c_r = self.current.angular.z # from odom
        t_r = self.wrap_to_2pi(self.target.angular.z) # target angle

        twist = Twist()
        if self.state == State.STOP: # direct stop
            self.current.linear.x = 0
            self.target.linear.x = 0

            self.target.angular.z = 0
            twist.linear.x = 0
            twist.angular.z = 0
        else:
            # linear update (Acc control)
            twist.linear.x = c_x + (t_x - c_x) * self.linear_p
            self.current.linear.x = c_x + (t_x - c_x) * self.linear_p
        
            # angular update
            twist.angular.z = (t_r - c_r) * self.rotate_p

        self.cmd_vel_pub.publish(twist)

    def set_timer(self, target):
        self.timer_target = target
        self.timer = 0
    
    def check_timer(self):
        if self.timer > self.target_target:
            return True
        else:
            self.timer += 1 / 30
            return False

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.follow()
            rate.sleep()


if __name__ == "__main__":
    node = Mover()
    node.run()
    rospy.spin()