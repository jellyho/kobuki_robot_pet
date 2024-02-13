import roslib; roslib.load_manifest('kobuki_testsuite')
import rospy
import sys

from tf.transformations import euler_from_quaternion
from math import degrees

from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion


class Mover:
    def __init__(self):
        rospy.init_node("mover")
        
        if (len(sys.argv) > 1):
            multip = int(sys.argv[1])
        else:
            multip = 1
        self.count = 0
        self.int_err = 0.0
        self.avg_err = 0.0
        self.target_v = 0.09*abs(multip)
        self.target_w = 0.1*multip
        self.twist = Twist()
        self.cmd_vel_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist)
        self.int_err_pub = rospy.Publisher("/int_err", Float32)
        self.avg_err_pub = rospy.Publisher("/avg_err", Float32)

        rospy.Subscriber("/mobile_base/sensors/imu_data", Imu, self.ImuCallback)
        rospy.Subscriber("/odom", Odometry, self.OdomCallback)
      
    def OdomCallback(self, data):
        self.odom = data.pose.pose.position.x
    
    def ImuCallback(self, data):
        self.count = self.count + 1
        if (self.count > 100):
            self.real_w = data.angular_velocity.z
            self.int_err += abs(self.target_w - self.real_w)
            self.avg_err =  self.int_err/(self.count - 100)
            
            self.int_err_pub.publish(self.int_err)
            self.avg_err_pub.publish(self.avg_err)

        self.twist.linear.x = self.target_v
        self.twist.angular.z = self.target_w
        self.cmd_vel_pub.publish(self.twist)



node = Mover()

while not rospy.is_shutdown():
    rospy.spin()