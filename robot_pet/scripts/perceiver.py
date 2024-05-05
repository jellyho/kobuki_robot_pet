#!/usr/bin/env python3

import roslib; roslib.load_manifest('kobuki_testsuite')
import rospy, rospkg
from sensor_msgs.msg import Imu, Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32, Int32, String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import sys, os, math, random, json, time, cv2
import numpy as np
import pandas as pd
import torch, time

rospack = rospkg.RosPack()
FacePath = rospack.get_path('robot_pet') + f'/face-recognition/'
os.chdir(FacePath)
sys.path.append(os.path.abspath(FacePath))

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Percevier:
    def __init__(self):
        rospy.init_node('perceiver', anonymous=True)
        self.command_pub = rospy.Publisher("/internal_command", Int32, queue_size=1)
        self.target_pub = rospy.Publisher("/target", Twist, queue_size=1)
        self.prcessed_pub = rospy.Publisher('/robot/image', Image, queue_size=1)
        # self.pc_pub = rospy.Publisher("/person_pc", PointCloud2, queue_size=1)

        self.datas = {
            "raw_image": [],
            "pose_keypoints": [],
            "pose_box":[],
            "pose_class":[],
            "id_face_mapping": {},
            "face_bboxes": [],
            "face_landmarks": [],
            "face_tracking_ids": [],
            "face_tracking_bboxes": [],
            "face_tracking_tlwhs": [],
            "ball_detections": []
        }

        self.model_weight_dir = rospack.get_path('robot_pet') + '/weight/model.h5'
        self.pose_model, _ = utils.load_model_ext(self.model_weight_dir)
        # Face detector (choose one)
        self.face_detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

        # face_detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
        self.tracker_config = utils.load_config("./face_tracking/config/config_tracking.yaml")
        self.tracker = BYTETracker(args=self.tracker_config, frame_rate=30)

        # Face recognizer
        self.face_recognizer = iresnet_inference(model_name="r18", path="face_recognition/arcface/weights/arcface_r18.pth", device=device)

        # Load precomputed face features and names
        self.images_names, self.images_embs = read_features(feature_path="./datasets/face_features/feature")

        self.image_sub = rospy.Subscriber(f'/camera/color/image_raw', Image, self.image_subscriber)
        # self.yolo_image_sub = rospy.Subscriber('/yolo_image', Image, self.yolo_image_cb)
        # self.pc_sub = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.pc_cb)
        self.depth_image_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
        self.pose_sub = rospy.Subscriber('/pose/yolo_results', String, callback=self.yolo_pose_cb)
        self.detection_sub = rospy.Subscriber('/detection/yolo_results', String, callback=self.yolo_detection_cb)
        self.camera_info = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_cb)

        self.pose = None
        self.detection = None
        self.depth = None
        self.image = None
        self.pc = None
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0

        self.person_th = 0.5

    def face_detect(self):
        try:
            current_img = self.datas['raw_image']
            outputs, img_info, bboxes, landmarks = self.face_detector.detect_tracking(image=current_img)

            tracking_tlwhs = []
            tracking_ids = []
            tracking_scores = []
            tracking_bboxes = []

            if outputs is not None:
                online_targets = self.tracker.update(outputs, [img_info["height"], img_info["width"]], (128, 128))
                for i in range(len(online_targets)):
                    t = online_targets[i]
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > self.tracker_config["aspect_ratio_thresh"]
                    if tlwh[2] * tlwh[3] > self.tracker_config["min_box_area"] and not vertical:
                        x1, y1, w, h = tlwh
                        tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                        tracking_tlwhs.append(tlwh)
                        tracking_ids.append(tid)
                        tracking_scores.append(t.score)

            self.datas["face_bboxes"] = bboxes
            self.datas["face_landmarks"] = landmarks
            self.datas['face_tracking_tlwhs'] = tracking_tlwhs
            self.datas["face_tracking_ids"] = tracking_ids
            self.datas["face_tracking_bboxes"] = tracking_bboxes

            detection_landmarks = self.datas["face_landmarks"]
            detection_bboxes = self.datas["face_bboxes"]
            tracking_ids = self.datas["face_tracking_ids"]
            tracking_bboxes = self.datas["face_tracking_bboxes"]
            try:
                for i in range(len(tracking_bboxes)):
                    for j in range(len(detection_bboxes)):
                        mapping_score = utils.mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                        if mapping_score > 0.9:
                            face_alignment = norm_crop(img=current_img, landmark=detection_landmarks[j])

                            # Get feature from face
                            face_image = utils.preprocess(face_alignment)
                            emb_img_face = self.face_recognizer(face_image.to(device)).detach().cpu().numpy()
                            query_emb = emb_img_face / np.linalg.norm(emb_img_face)

                            score, id_min = compare_encodings(query_emb, self.images_embs)
                            name = self.images_names[id_min]
                            score = score[0]
                            if name is not None:
                                if score < 0.25:
                                    caption = "UN_KNOWN"
                                else:
                                    caption = f"{name}:{score:.2f}"
                            self.datas['id_face_mapping'][tracking_ids[i]] = caption
                            detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                            detection_landmarks = np.delete(detection_landmarks, j, axis=0)
                            break
            except Exception as e:
                print("Error in recognizing faces")
                print(e)
        except CvBridgeError as e:
            rospy.logerr(e)        

    def image_subscriber(self, image_msg):
        bridge = CvBridge()
        self.datas['raw_image'] = bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")

        processed_img = utils.plot_tracking(self.datas['raw_image'],
                                    self.datas['pose_keypoints'],
                                    self.datas['pose_box'],
                                    self.datas['pose_class'],
                                    self.datas['ball_detections'],
                                    self.datas['face_tracking_tlwhs'],
                                    self.datas['face_tracking_ids'],
                                    names = self.datas["id_face_mapping"],
                                    frame_id = 1,
                                    fps = 0.0
        )
        ros_image = bridge.cv2_to_imgmsg(processed_img, encoding="rgb8")
        self.prcessed_pub.publish(ros_image)

    def info_cb(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]

    def yolo_pose_cb(self, msg):
        if msg:
            self.pose = json.loads(msg.data)
        

    def pose_classification(self):
        result = self.pose
        if len(result) > 0:
            pose = result[0]['keypoints']
            box = result[0]['box']

            class_names = ['Stop', 'Start', 'Unknown', 'Unknown', 'Unknown']
            col_names = [
                '0_X', '0_Y', '1_X', '1_Y', '2_X', '2_Y', '3_X', '3_Y', '4_X', '4_Y', '5_X', '5_Y', 
                '6_X', '6_Y', '7_X', '7_Y', '8_X', '8_Y', '9_X', '9_Y', '10_X', '10_Y', '11_X', '11_Y', 
                '12_X', '12_Y', '13_X', '13_Y', '14_X', '14_Y', '15_X', '15_Y', '16_X', '16_Y'
            ]

            x, y, conf = pose['x'], pose['y'], pose['visible']
            lm_list = [[int(xx), int(yy)] for xx, yy in zip(x, y)]

            # if np.sum(np.array(conf) > 0.1) >= 17:
            if True:
                pre_lm = utils.norm_kpts(lm_list)
                data = pd.DataFrame([pre_lm], columns=col_names)
                predict = self.pose_model.predict(data, verbose=0)[0]

                if max(predict) > 0.6:
                    pose_class = class_names[predict.argmax()]
                else:
                    pose_class = 'Unknown Pose'
                self.datas['pose_box'] = box
                self.datas['pose_keypoints'] = lm_list
                self.datas['pose_class'] = pose_class
        else:
            self.datas['pose_box'] = {}
            self.datas['pose_keypoints'] = {}
            self.datas['pose_class'] = {}

    def yolo_detection_cb(self, msg):
        result = json.loads(msg.data)

        desired_class_ids = list(range(32, 34))  # 30, 31, 32, 33에 해당하는 객체만 선택
        
        for box in result:
            if box['class'] in desired_class_ids:
                bbox = box['box']
                box_width_px = bbox['x2'] + bbox['x1']

                distance_cm = utils.estimate_distance(box_width_px)
                self.datas['ball_detections'] = {'box':bbox, 'distance':distance_cm}
            else:
                self.datas['ball_detections'] = {}

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
            if isinstance(self.datas['raw_image'], np.ndarray):
                self.face_detect()

            if self.pose is None:
                # send zero target
                twist = Twist()
                twist.angular.z = 0
                self.target_pub.publish(twist)
                # No person. STOP
                self.command_pub.publish(0)
                continue
            else:
                self.pose_classification()
                valid_people = []
                # append real people
                for person in self.pose:
                    if person['confidence'] > self.person_th:
                        valid_people.append(person)

                # simple following
                if len(valid_people) > 0 and self.image is not None and self.depth is not None:
                    target = valid_people[0]
                    person_center = (target['box']['x1'] + target['box']['x2']) / 2, (target['box']['y1'] + target['box']['y2']) / 2
                    distance = None

                    twist = Twist()
                    twist.angular.z = -(person_center[0] / self.image.shape[1] - 0.5)
                    self.target_pub.publish(twist)
                    self.command_pub.publish(1) # Follow
                else:
                    self.command_pub.publish(0)
                
            key = self.getKey()
            if key:
                rospy.loginfo("Current pose: {}".format(key))
                if key in [1, 2, 3, 4]:
                    self.command_pub.publish(1) # Follow
                else:
                    self.command_pub.publish(0) # Stop



if __name__ == "__main__":
    node = Percevier()
    node.run()
    cv2.destroyAllWindows()
    print('[INFO] Inference on Videostream is Ended...')
    rospy.spin()