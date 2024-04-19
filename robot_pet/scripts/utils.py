from keras.models import load_model
from keras.models import save_model
import yaml
import torch
from torchvision import transforms
import numpy as np
import cv2
import h5py
import math
import random

def sign(x):
    if x > 0: return +1
    if x < 0: return -1
    return 0

def load_config(file_name):
    """
    Load a YAML configuration file.

    Args:
        file_name (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

@torch.no_grad()
def preprocess(face_image):
    """
    Extract features from a face image.

    Args:
        face_image: The input face image.

    Returns:
        numpy.ndarray: The extracted features.
    """
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image (BGR)
    face_image = face_preprocess(face_image).unsqueeze(0)

    return face_image


def mapping_bbox(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
        box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoU score.
    """
    # Calculate the intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color
def plot_tracking(
    image, pose_detection, ball_detection, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None, names=[]
):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    if ball_detection and len(ball_detection) > 0:
        plot_one_box(list(ball_detection['box'].values()), im, label=f"ball:{ball_detection['distance']:2f}m")

    radius = max(5, int(im_w / 140.0))
    cv2.putText(
        im,
        "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 255),
        thickness=2,
    )
    if tlwhs:
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "{}".format(int(obj_id))
            if (obj_id) in names:
                id_text = id_text + ": " + names[obj_id]
            if ids2 is not None:
                id_text = id_text + ", {}".format(int(ids2[i]))
            color = get_color(abs(obj_id))
            cv2.rectangle(
                im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
            )
            cv2.putText(
                im,
                id_text,
                (intbox[0], intbox[1]),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 0, 255),
                thickness=text_thickness,
            )


    # for result in pose_detection:
    #     result.orig_img = im
    #     im = result.plot()

    #     # print(result.keypoints.data.detach().cpu().numpy())
    #     #print(result.keypoints.data)
    #     # cv2.drawKeypoints(im, result.keypoints.data)

    #     #for result in object_detection:
    #     #    print(result.keypoints.data)

    # for result in object_detection:
    #     result.orig_img = im
    #     im = result.plot()


    return im

def plot_skeleton_kpts(im, kpts, radius=5, shape=(640, 640), confi=0.5, line_thick=2):
    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                            dtype=np.uint8)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    
    limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    ndim = kpts.shape[-1]
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]]
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < confi:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    for i, sk in enumerate(skeleton):
        pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
        pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
        if ndim == 3:
            conf1 = kpts[(sk[0] - 1), 2]
            conf2 = kpts[(sk[1] - 1), 2]
            if conf1 < confi or conf2 < confi:
                continue
        if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, [int(x) for x in limb_color[i]], thickness=line_thick, lineType=cv2.LINE_AA)


# Normalize Keypoints
def norm_kpts(lm_list, torso_size_multiplier=2.5):
    max_distance = 0
    center_x = (lm_list[12][0] +       # right_hip
                lm_list[11][0])*0.5    # left_hip
    center_y = (lm_list[12][1] +       # right_hip
                lm_list[11][1])*0.5    # left_hip

    shoulders_x = (lm_list[6][0] +       # right_shoulder
                    lm_list[5][0])*0.5   # left_shoulder
    shoulders_y = (lm_list[6][1] +       # right_shoulder
                    lm_list[5][1])*0.5   # left_shoulder
    
    if max_distance == 0:
        # max_distance가 0이면 경고를 출력하고 기본 값 설정
        print("Warning: max_distance is 0, setting to default value to avoid division by zero.")
        max_distance = 1  # 기본값 설정 (예: 1)
        
    for lm in lm_list:
        distance = math.sqrt(
            (lm[0] - center_x)**2 + (lm[1] - center_y)**2)
        if(distance > max_distance):
            max_distance = distance
    torso_size = math.sqrt(
        (shoulders_x - center_x)**2 + (shoulders_y - center_y)**2)
    max_distance = max(
        torso_size*torso_size_multiplier, max_distance)

    pre_lm = list(np.array(
        [[(landmark[0]-center_x)/max_distance, (landmark[1]-center_y)/max_distance] for landmark in lm_list]
    ).flatten())
    

    
    return pre_lm


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# Save and Load Keras Model with meta data
def load_model_ext(filepath):
    model = load_model(filepath, custom_objects=None, compile=False)
    f = h5py.File(filepath, mode='r')
    meta_data = None
    if 'my_meta_data' in f.attrs:
        meta_data = f.attrs.get('my_meta_data')
    f.close()
    return model, meta_data
   

def save_model_ext(model, filepath, overwrite=True, meta_data=None):
    save_model(model, filepath, overwrite)
    if meta_data is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['my_meta_data'] = meta_data
        f.close()

def estimate_distance(ball_width_px, real_diameter_cm=6):
    # 가상의 초점 거리 (픽셀 단위)
    focal_length_px = 800  # 이 값은 카메라 보정을 통해 얻어야 합니다.
    # 상대적 거리 추정
    distance_cm = (real_diameter_cm * focal_length_px) / ball_width_px
    return distance_cm