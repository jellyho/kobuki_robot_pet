#!/usr/bin/env python3

import roslib; roslib.load_manifest('kobuki_testsuite')
import rospy, sys, random, rospkg

from tf.transformations import euler_from_quaternion
from math import degrees

from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from kobuki_msgs.msg import BumperEvent, CliffEvent
from enum import Enum

import sys, select, termios, tty

import os
from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math

##############
class Percevier:
    def __init__(self):
        rospy.init_node('perceiver', anonymous=True)
        self.command_pub = rospy.Publisher("/internal_command", Int32, queue_size=1)
        self.target_pub = rospy.Publisher("/target", Twist, queue_size=1)
        self.rospack = rospkg.RosPack()
        self.model_weight_dir = self.rospack.get_path('robot_pet') + '/weight/model2.h5'
        self.torso_size_multiplier = 2.5
        n_landmarks = 33
        self.n_dimensions = 3
        self.threshold = 0.6
        self.landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]
        self.class_names = [
            'Right', 'Left', 'Foward',
            'Backward'
        ]
        ##############

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.col_names = []
        for i in range(n_landmarks):
            name = self.mp_pose.PoseLandmark(i).name
            name_x = name + '_X'
            name_y = name + '_Y'
            name_z = name + '_Z'
            name_v = name + '_V'
            self.col_names.append(name_x)
            self.col_names.append(name_y)
            self.col_names.append(name_z)
            self.col_names.append(name_v)

        # Load saved model
        self.model = load_model(self.model_weight_dir, compile=False)

        # Web-cam
        self.cap = cv2.VideoCapture(0)
        self.source_width = int(self.cap.get(3))
        self.source_height = int(self.cap.get(4))

    def getKey(self):
        success, img = self.cap.read()
        if not success:
            print('[ERROR] Failed to Read Video feed')
            return 0
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(img_rgb)
        key = 0
        if result.pose_landmarks:
            lm_list = []
            for landmarks in result.pose_landmarks.landmark:
                # Preprocessing
                max_distance = 0
                lm_list.append(landmarks)
            center_x = (lm_list[self.landmark_names.index('right_hip')].x +
                        lm_list[self.landmark_names.index('left_hip')].x)*0.5
            center_y = (lm_list[self.landmark_names.index('right_hip')].y +
                        lm_list[self.landmark_names.index('left_hip')].y)*0.5

            shoulders_x = (lm_list[self.landmark_names.index('right_shoulder')].x +
                            lm_list[self.landmark_names.index('left_shoulder')].x)*0.5
            shoulders_y = (lm_list[self.landmark_names.index('right_shoulder')].y +
                            lm_list[self.landmark_names.index('left_shoulder')].y)*0.5

            for lm in lm_list:
                distance = math.sqrt((lm.x - center_x) **
                                        2 + (lm.y - center_y)**2)
                if(distance > max_distance):
                    max_distance = distance
            torso_size = math.sqrt((shoulders_x - center_x) **
                                    2 + (shoulders_y - center_y)**2)
            max_distance = max(torso_size*self.torso_size_multiplier, max_distance)

            pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                                        landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())
            data = pd.DataFrame([pre_lm], columns=self.col_names)
            predict = self.model.predict(data, verbose=0)[0]
            if max(predict) > self.threshold:
                key = predict.argmax()
                pose_class = self.class_names[key]
                img = cv2.circle(img, (int((center_x)*self.source_width) ,int(self.source_height / 2)), 10, (0, 255, 0), -1)
                # publish target
                twist = Twist()
                rospy.loginfo(center_x)
                # twist.linear.x = (center_x - self.source_width) / self.source_width
                twist.linear.x = -(center_x - 0.5)
                twist.linear.y = 0
                twist.linear.z = 0
                twist.angular.x = 0
                twist.angular.y = 0
                twist.angular.z = 0
                self.target_pub.publish(twist)
            else:
                pose_class = 'Unknown Pose'
                twist = Twist()
                rospy.loginfo(center_x)
                # twist.linear.x = (center_x - self.source_width) / self.source_width
                twist.linear.x = 0
                twist.linear.y = 0
                twist.linear.z = 0
                twist.angular.x = 0
                twist.angular.y = 0
                twist.angular.z = 0
                self.target_pub.publish(twist)
            # Show Result
            img = cv2.putText(
                img, f'{pose_class}',
                (40, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2
            )

            

        cv2.imshow('Output Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        
        

        return key + 2

    def run(self):
        while not rospy.is_shutdown():
            key = self.getKey()
            if key:
                rospy.loginfo("Current pose: {}".format(key))
                key = 1 # Follow
                self.command_pub.publish(key)



if __name__ == "__main__":
    node = Percevier()
    node.run()
    node.cap.release()
    cv2.destroyAllWindows()
    print('[INFO] Inference on Videostream is Ended...')
    rospy.spin()