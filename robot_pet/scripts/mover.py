#!/usr/bin/env python3

import roslib; roslib.load_manifest('kobuki_testsuite')
import rospy, sys, random

from tf.transformations import euler_from_quaternion
from math import degrees

from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import WheelDropEvent
from kobuki_msgs.msg import BumperEvent, CliffEvent
import numpy as np
from enum import Enum

class State(Enum):
    STOP = 0
    FOLLOW = 1
    COMMAND = 2
    WAIT = 3
    EMERGENCY = 4
    EMER_BACK = 5


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
        self.yaw_offset = 0
        self.state = State.STOP

        # Data for movement
        self.rotate_p = 1.3
        self.linear_p = 0.2
        self.wandering_offset = 0.3 # 0.5pi = 90degrees
        self.follow_x = True
        self.follow_z = True
        self.timer_set = False
        self.timer_target = 0.0
        self.timer = 0.0

        # Subscriber
        self.bumper_sub = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bumper_cb)
        self.cliff_sub = rospy.Subscriber('/mobile_base/events/cliff', CliffEvent, self.cliff_cb)
        self.drop_sub = rospy.Subscriber("/mobile_base/events/wheel_drop",WheelDropEvent, self.drop_cb)


        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.command_sub = rospy.Subscriber("/internal_command", Int32, self.command_cb)
        self.target_sub = rospy.Subscriber("/target", Twist, self.target_cb)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=1)

    def target_cb(self, data): # following target
        if self.state == State.FOLLOW:
            self.target.angular.z = self.current.angular.z + data.angular.z # update target

    def odom_cb(self, data):
        self.odom = data
        quat = data.pose.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        if np.abs(self.current.angular.z - self.yaw_offset - yaw) > np.pi:
            if self.current.angular.z - self.yaw_offset > yaw:
                self.yaw_offset += 2 * np.pi
            elif self.current.angular.z - self.yaw_offset < yaw:
                self.yaw_offset -= 2 * np.pi
            # rospy.loginfo(f'OVERFLOW!! {np.abs(self.current.angular.z - yaw)}, Prev{self.current.angular.z}, Now{yaw}, Offset{self.yaw_offset}')
        self.current.angular.z = yaw + self.yaw_offset
      
    def cliff_cb(self, data):
        self.cliff = data
        if data.state == CliffEvent.CLIFF and self.state != State.EMERGENCY:
            self.state = State.EMERGENCY
            self.target.linear.x = 0
            if data.sensor == CliffEvent.LEFT:
                self.set_target(r=self.current.angular.z - 3.141592 * self.wandering_offset)
            elif data.sensor == CliffEvent.RIGHT:
                self.set_target(r=self.current.angular.z + 3.141592 * self.wandering_offset)
            else:
                self.set_target(r=self.current.angular.z + 3.141592 * (random.randint(0, 1)-0.5))
            self.set_target(x=-0.2, follow_z=False)
            self.set_timer(target=2)

    def bumper_cb(self, data):
        self.bumper = data
        if data.state == BumperEvent.PRESSED and self.state != State.EMERGENCY:
            self.state = State.EMERGENCY
            self.target.linear.x = 0
            if data.bumper == BumperEvent.LEFT:
                self.set_target(r=self.current.angular.z - 3.141592 * self.wandering_offset)
            elif data.bumper == BumperEvent.CENTER:
                self.set_target(r=self.current.angular.z + 3.141592 * (random.randint(0, 1)-0.5))
            else:
                self.set_target(r=self.current.angular.z + 3.141592 * self.wandering_offset)
            self.set_target(x=-0.2, follow_z=False)
            self.set_timer(target=2)
                

    def drop_cb(self, data):
        if data.state != WheelDropEvent.RAISED:
            rospy.signal_shutdown("Robot Wheel Raised... Shutdown")

    def set_target(self, x=None, r=None, follow_x=True, follow_z=True):
        self.follow_x = follow_x
        self.follow_z = follow_z
        self.target.linear.x = x if x is not None else self.target.linear.x
        self.target.angular.z = r if r is not None else self.target.angular.z
        # rospy.loginfo(f'Set Target x:{self.target.linear.x}, r:{self.target.angular.z}')

    def action_spin(self):
        self.set_target(0, self.current.angular.z + np.pi * 2)
        self.set_timer(4)

    def action_foward(self):
        speed = 0.2
        self.set_target(speed, self.current.angular.z)
        self.set_timer(0.05)

    def action_backward(self):
        speed = 0.2
        self.set_target(-speed, self.current.angular.z)
        self.set_timer(0.05)

    def action_turn_right(self):
        speed = 1
        self.set_target(0, self.current.angular.z - speed)
        self.set_timer(0.05)

    def action_turn_left(self):
        speed = 1
        self.set_target(0, self.current.angular.z + speed)
        self.set_timer(0.05)

    def command_cb(self, data):
        self.command = data.data

    def fsm(self):
        # implement FSM
        if self.state == State.STOP or self.state == State.COMMAND:
            if self.state == State.STOP: #STOP
                self.set_target(0, self.current.angular.z)
            if self.command == 0:
                self.state = State.STOP
            elif self.command == 1:
                self.state = State.FOLLOW
                self.set_target(x=0.1)
            elif self.command >= 2:
                if self.command == 2:
                    self.action_spin()
                ########### keyop ##################
                elif self.command == 10:
                    self.action_foward()
                elif self.command == 11:
                    self.action_turn_left()
                elif self.command == 12:
                    self.action_backward()
                elif self.command == 13:
                    self.action_turn_right()
                ########### keyop end ###############
                self.state = State.WAIT
        elif self.state == State.EMERGENCY: #EMERGENCY
            if self.check_timer():
                self.set_target(x=0)
                self.set_timer(2)
                self.state = State.WAIT
        elif self.state == State.WAIT: #WAIT
            if self.check_timer():
                self.state = State.COMMAND
        elif self.state == State.FOLLOW:
            if self.command != 1:
                self.state = State.COMMAND

    def follow(self):
        # load current
        c_x = self.current.linear.x # curretn velocity
        t_x = self.target.linear.x # target velocity

        c_r = self.current.angular.z # from odom
        t_r = self.target.angular.z # target angle

        twist = Twist()
        if self.state == State.STOP: # direct stop
            self.current.linear.x = 0
            self.set_target(0, 0)

            twist.linear.x = 0
            twist.angular.z = 0
        else:
            # linear update (Acc control)
            if not self.follow_x:
                t_x = 0
            twist.linear.x = c_x + (t_x - c_x) * self.linear_p
            self.current.linear.x = c_x + (t_x - c_x) * self.linear_p
        
            # angular update
            if not self.follow_z:
                twist.angular.z = 0
            else:
                twist.angular.z = (t_r - c_r) * self.rotate_p

        self.cmd_vel_pub.publish(twist)

    def set_timer(self, target):
        # rospy.loginfo(f'SET TIMER {target}')
        self.timer_target = target
        self.timer = 0
        self.timer_set = True
    
    def check_timer(self):
        if self.timer_set and self.timer > self.timer_target:
            self.timer = 0
            self.timer_target = 0
            self.timer_set = False
            return True
        else:
            self.timer += 1 / 30
            return False

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.fsm()
            self.follow()
            rate.sleep()


if __name__ == "__main__":
    node = Mover()
    node.run()
    rospy.spin()