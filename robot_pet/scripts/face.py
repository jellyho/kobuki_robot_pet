#!/usr/bin/env python3

import threading
import time, json, os, sys

import rospy, rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch
from torchvision import transforms

rospack = rospkg.RosPack()
FacePath = rospack.get_path('robot_pet') + f'/face_recognition/'
os.chdir(FacePath)
sys.path.append(os.path.abspath(FacePath))

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
import utils

from face_recognition.adaface.model import adaface_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Face:
    def __init__(self):
        rospy.init_node('face_node', anonymous=True)

        self.datas = {
        "id_face_mapping": {},
        "face_bboxes": [],
        "face_landmarks": [],
        "face_tracking_ids": [],
        "face_tracking_bboxes": [],
        "face_tracking_tlwhs": []
        }

        self.json_publisher = rospy.Publisher(f'/face/face_results', String, queue_size=1)
        rospy.Subscriber(f'/camera/color/image_raw', Image, self.image_subscriber)

        # Face detector (choose one)
        self.face_detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

        # face_detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
        self.tracker_config = utils.load_config("./face_tracking/config/config_tracking.yaml")
        self.tracker = BYTETracker(args=self.tracker_config, frame_rate=30)

        # Face recognizer
        #self.face_recognizer = iresnet_inference(model_name="r34", path="face_recognition/arcface/weights/arcface_r34.pth", device=device)
        self.face_recognizer = adaface_inference(model_name='r50', path='face_recognition/adaface/weights/adaface_ir50_webface4m.ckpt', device=device)
        
        # Load precomputed face features and names
        self.images_names, self.images_embs = read_features(feature_path="./datasets/face_features/feature")
        
    def image_subscriber(self, image_msg):
        bridge = CvBridge()
        t = rospy.Time.now()
        try:
            frame = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            current_img = frame
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

            self.datas["face_bboxes"] = list(bboxes[0])
            self.datas["face_landmarks"] = landmarks
            self.datas['face_tracking_tlwhs'] = tracking_tlwhs
            self.datas["face_tracking_ids"] = tracking_ids
            self.datas["face_tracking_bboxes"] = tracking_bboxes

            detection_landmarks = self.datas["face_landmarks"]
            detection_bboxes = self.datas["face_bboxes"]
            tracking_ids = self.datas["face_tracking_ids"]
            tracking_bboxes = self.datas["face_tracking_bboxes"]
            s1 = rospy.Time.now() - t
            try:
                for i in range(len(tracking_bboxes)):
                    for j in range(len(detection_bboxes)):
                        mapping_score = utils.mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                        if mapping_score > 0.9:
                            face_alignment = norm_crop(img=current_img, landmark=detection_landmarks[j])

                            # Get feature from face
                            face_image = utils.preprocess(face_alignment,type= 'bgr')
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
            
            # self.json_publisher.publish(String(json.dumps(self.datas)))
            print(self.datas)

        except CvBridgeError as e:
            rospy.logerr(e)
            

if __name__ == '__main__':
    try:
        Face()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
