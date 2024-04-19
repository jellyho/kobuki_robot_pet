import threading
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker

from ultralytics import YOLO
import jung_utils

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datas = {
    "raw_image": [],

    "pose_detections": [],
    "master_pose": [],

    'object_detections': [],

    "id_face_mapping": {},
    "face_bboxes": [],
    "face_landmarks": [],
    "face_tracking_ids": [],
    "face_tracking_bboxes": [],
    "face_tracking_tlwhs": []
}

def read_frames():
    # Initialize variables for measuring frame rate
    start_time = time.time_ns()
    frame_count = 0
    fps = -1

    # Initialize a tracker and a timer
    frame_id = 0

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        if not ret:
            break

        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

        datas['raw_image'] = img

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()


        try:
            datas['master_pose'] = []
            for i, face_id in enumerate(datas['face_tracking_ids']):
                face_id = int(face_id)
                if face_id in datas["id_face_mapping"] and datas["id_face_mapping"][face_id] != 'UN_KNOWN':

                    # find matching keypoint
                    for pose in datas['pose_detections'][0]:
                        keypoints = pose.keypoints.xy.squeeze()
                        nose = keypoints[0]
                        # left_eye = keypoints[1]
                        # right_eye = keypoints[2]
                        x, y, w, h = datas['face_tracking_tlwhs'][i]
                        box_coords = (x, y, x + w, y + h)

                        if jung_utils.is_inside_box(nose, box_coords):
                            datas['master_pose'].append(pose)
        except Exception as e:
            print(e)

        try:
            img = jung_utils.plot_tracking(datas['raw_image'],
                                        datas['master_pose'],
                                        datas['object_detections'],
                                        datas['face_tracking_tlwhs'],
                                        datas['face_tracking_ids'],
                                        names = datas["id_face_mapping"],
                                        frame_id = frame_id + 1,
                                        fps = fps
            )

        except Exception as e:
            print(e)

        cv2.imshow("Yonsei Graduation", img)

def detect_faces():

    # Face detector (choose one)
    face_detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

    # face_detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
    tracker_config = jung_utils.load_config("./face_tracking/config/config_tracking.yaml")
    tracker = BYTETracker(args= tracker_config, frame_rate=30)

    # Face recognizer
    face_recognizer = iresnet_inference(
        model_name="r34", path="face_recognition/arcface/weights/arcface_r34.pth", device=device
    )

    # Load precomputed face features and names
    images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

    while True:
        try:
            current_img = datas['raw_image']
            outputs, img_info, bboxes, landmarks = face_detector.detect_tracking(image= current_img)

            tracking_tlwhs = []
            tracking_ids = []
            tracking_scores = []
            tracking_bboxes = []

            if outputs is not None:
                online_targets = tracker.update(
                    outputs, [img_info["height"], img_info["width"]], (128, 128)
                )

                for i in range(len(online_targets)):
                    t = online_targets[i]
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > tracker_config["aspect_ratio_thresh"]
                    if tlwh[2] * tlwh[3] > tracker_config["min_box_area"] and not vertical:
                        x1, y1, w, h = tlwh
                        tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                        tracking_tlwhs.append(tlwh)
                        tracking_ids.append(tid)
                        tracking_scores.append(t.score)

                datas['face_tracking_tlwhs'] = tracking_tlwhs
                datas['face_tracking_ids'] = tracking_ids

                for i in range(len(tracking_bboxes)):
                    for j in range(len(bboxes)):
                        mapping_score = jung_utils.mapping_bbox(box1=tracking_bboxes[i], box2=bboxes[j])
                        if mapping_score > 0.9:
                            face_alignment = norm_crop(img= current_img, landmark= landmarks[j])

                            # Get feature from face
                            face_image = jung_utils.preprocess(face_alignment)
                            emb_img_face = face_recognizer(face_image.to(device)).detach().cpu().numpy()
                            query_emb = emb_img_face / np.linalg.norm(emb_img_face)

                            score, id_min = compare_encodings(query_emb, images_embs)
                            name = images_names[id_min]
                            score = score[0]

                            if name is not None:
                                if score < 0.25:
                                    caption = "UN_KNOWN"
                                else:
                                    caption = f"{name}:{score:.2f}"
                            datas['id_face_mapping'][tracking_ids[i]] = caption
                            bboxes = np.delete(bboxes, j, axis=0)
                            landmarks = np.delete(landmarks, j, axis=0)

        except Exception as e:
            print("Error in detecting faces")
            print(e)

def detect_poses():
    # Pose detection
    pose_detector = YOLO('yolov8n-pose.pt').to(device)

    while True:
        try:
            current_img = datas['raw_image']
            # results = pose_detector.predict(current_img)
            results = pose_detector.predict(current_img, stream_buffer = True , verbose=False)
            # print(results[0].keypoints.data)
            datas['pose_detections'] = results

        except Exception as e:
            print("Error in detecting poses")
            print(e)

def detect_objects():
    object_detector = YOLO("yolov8n.pt").to(device)
    while True:
        try:
            current_img = datas['raw_image']
            results = object_detector.predict(current_img, stream_buffer=True, verbose=False)
            datas['object_detections'] = results

        except Exception as e:
            print("Error in detecting objects")
            print(e)


if __name__ == "__main__":

    thread_frames = threading.Thread(target= read_frames)
    thread_detect_faces = threading.Thread(target= detect_faces)
    thread_detect_poses = threading.Thread(target= detect_poses)
    thread_detect_objects = threading.Thread(target= detect_objects)

    thread_frames.start()
    thread_detect_faces.start()
    thread_detect_poses.start()
    thread_detect_objects.start()