#!/usr/bin/env python3

import roslib; roslib.load_manifest('kobuki_testsuite')
import rospy, rospkg
from sensor_msgs.msg import Imu, Image
from std_msgs.msg import Float32, Int32, String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import sys, os, math, random, json
import cv2
import numpy as np
import pandas as pd

class Percevier:
    def __init__(self):
        rospy.init_node('perceiver', anonymous=True)
        self.command_pub = rospy.Publisher("/internal_command", Int32, queue_size=1)
        self.target_pub = rospy.Publisher("/target", Twist, queue_size=1)
        self.rospack = rospkg.RosPack()
        self.model_weight_dir = self.rospack.get_path('robot_pet') + '/weight/model2.h5'
        self.yolo_image_sub = rospy.Subscriber('/yolo_image', Image, self.yolo_image_cb)
        self.depth_image_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
        self.pose_sub = rospy.Subscriber('/yolo_results', String, callback=self.yolo_cb)
        self.pose = None
        self.depth = None
        self.image = None
        self.person_th = 0.5

    def yolo_cb(self, msg):
        result = json.loads(msg.data)
        self.pose = result

    def depth_cb(self, msg):
        bridge = CvBridge()
        depth_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth = depth_frame

    def yolo_image_cb(self, msg):
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.image = frame

    def run(self):
        while not rospy.is_shutdown():
            if self.pose is None:
                # send zero target
                twist = Twist()
                twist.angular.z = 0
                self.target_pub.publish(twist)
                # No person. STOP
                self.command_pub.publish(0)
                continue
            else:
                valid_people = []
                # append real people
                for person in self.pose:
                    if person['confidence'] > self.person_th:
                        valid_people.append(person)

                # simple following
                if len(valid_people) > 0 and self.image is not None and self.depth is not None:
                    target = valid_people[0]
                    person_center = (target['box']['x1'] + target['box']['x2']) / 2, (target['box']['y1'] + target['box']['y2']) / 2
                    # distance ?
                    distance = None
                    if self.depth is not None:
                        mean_dst = np.mean(self.depth[int(target['box']['y1']):int(target['box']['y2'])][int(target['box']['x1']):int(target['box']['x2'])])
                        print(mean_dst)

                    twist = Twist()
                    twist.angular.z = -(person_center[0] / self.image.shape[1] - 0.5)
                    self.target_pub.publish(twist)
                    self.command_pub.publish(1) # Follow
                else:
                    self.command_pub.publish(0)
                
            # key = self.getKey()
            # if key:
            #     rospy.loginfo("Current pose: {}".format(key))
            #     if key in [1, 2, 3, 4]:
            #         self.command_pub.publish(1) # Follow
            #     else:
            #         self.command_pub.publish(0) # Stop



if __name__ == "__main__":
    node = Percevier()
    node.run()
    cv2.destroyAllWindows()
    print('[INFO] Inference on Videostream is Ended...')
    rospy.spin()