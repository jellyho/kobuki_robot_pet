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

import sys, select, termios, tty

class Percevier:
    def __init__(self):
        rospy.init_node('perceiver_keyop', anonymous=True)
        self.command_pub = rospy.Publisher("/internal_command", Int32)
        self.settings = termios.tcgetattr(sys.stdin)

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        try:
            while not rospy.is_shutdown():
                key = self.getKey()
                if key:
                    rospy.loginfo("Pressed key: {}".format(key))
                    # Convert the key to integer if needed
                    key_int = ord(key)  # Convert character to ASCII integer
                    # Publish the integer topic
                    if key_int == 119: # w
                        key_int = 1
                    else:
                        key_int = 0
                    self.command_pub.publish(key_int)
        except Exception as e:
            print(e)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


if __name__ == "__main__":
    node = Percevier()
    node.run()
    rospy.spin()