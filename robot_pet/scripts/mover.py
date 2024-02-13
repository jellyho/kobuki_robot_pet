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
from robot_pet import wrap_to_pi

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

        # Subscriber
        self.bumper_sub = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bumper_cb)
        self.cliff_sub = rospy.Subscriber('/mobile_base/events/cliff', CliffEvent, self.cliff_cb)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.imu_sub = rospy.Subscriber("/mobile_base/sensors/imu_data", Imu, self.imu_cb)
        self.command_sub = rospy.Subscriber("/internal_command", Int32, self.command_cb)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist)
    
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

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            
            rate.sleep()



node = Mover()
node.run()