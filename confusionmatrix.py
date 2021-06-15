import logging
import subprocess
import time
import os
import cv2
from uuid import uuid4
from cv2 import data
# import pandas as pd
from glob import glob
from tqdm import tqdm
import re
import numpy as np

idx_cls_dict = {0: "standing", 1: "sitting", 2: "sitting on bed", 3: "lying on bed",
                4: "lying on floor", 7: "no label", 5: "occupied_bed", 6: "empty_bed"}
cls_dict = {"standing": 0, "sitting": 1, "sitting on bed": 2, "lying on bed": 3,
            "lying on floor": 4, "no label": 7, "occupied_bed": 5, "empty_bed": 6}


class YoloV4():
    def __init__(self, darknet: str, config_file: str, data_file: str, weights_file: str):
        # Use the path to the darknet file I sent you
        self.darknet = darknet
        self.config_file = config_file
        self.data_file = data_file
        self.weights_file = weights_file
        self.first_time = True
       # self.thresh = "-"
        assert os.path.exists(self.config_file)
        assert os.path.exists(self.weights_file)
        self.pr = subprocess.Popen(
            [self.darknet, 'detector', 'test', self.data_file, self.config_file, self.weights_file, '-dont_show', '-ext_output', '-thresh', '0.5'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
        print(self.pr)

    def stop(self):
        self.pr.stdin.close()

    def process(self, camera_images, _ceil=True):
        detected_objects = []
        for cam_idx, camera_image in enumerate(camera_images):
            detections = []
            image_path = self.__preprocess(camera_image)
            output = ""
            self.pr.stdin.write(image_path + "\n")
            while True:
                c = self.pr.stdout.read(1)
                output += c
                if "Enter Image Path" in output:
                    if self.first_time:
                        output = ""
                        self.first_time = False
                        continue
                    break
            result = output.split("\n")
            for i in range(5, len(result)):
                data = result[i-1]
                label = data.split(":")[0]
                confidient = int(data.split(":")[1].split("%")[0])
                bbox = data.split("(")[1].split(")")[0]
                if _ceil:
                    left_x = int(re.search('left_x:(.*)top_y', bbox).group(1))
                    left_x = 0 if left_x < 0 else left_x
                    top_y = int(re.search('top_y:(.*)width', bbox).group(1))
                    top_y = 0 if top_y < 0 else top_y
                    width = int(re.search('width:(.*)height', bbox).group(1))
                    height = int(re.search('height:(.*)', bbox).group(1))
                else:
                    left_x = float(
                        re.search('left_x:(.*)top_y', bbox).group(1))
                    left_x = 0. if left_x < 0 else left_x
                    top_y = float(re.search('top_y:(.*)width', bbox).group(1))
                    top_y = 0. if top_y < 0 else top_y
                    width = float(re.search('width:(.*)height', bbox).group(1))
                    height = float(re.search('height:(.*)', bbox).group(1))

                detections.append(
                    (label, confidient, (left_x, top_y, width, height), cam_idx))

            detected_objects = detected_objects + \
                self.__postprocess(camera_image, detections)
            os.remove(image_path)
        return detected_objects

    def __preprocess(self, camera_image):
        path = os.getcwd()
        image_path = path + "/" + uuid4().hex + ".jpg"
        cv2.imwrite(image_path, camera_image)
        return image_path

    def __postprocess(self, camera_image, detections):
        detected_objects = []
        for detection in detections:
            label, confidence, (x, y, w, h), cam_id = detection
            detected_objects.append([
                camera_image,       # CameraImage
                label,              # str
                (x, y, w, h),       # BoundingBox
                confidence,         # int
                cam_id
            ])
        return detected_objects


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def draw(img, bboxes,color):
    if color ==1:
        for bb in bboxes:
            cv2.rectangle(img, (bb["x1"], bb["y1"]), (bb["x2"], bb["y2"]), (0, 255, 0), 1)
            cv2.putText(img, str(
                idx_cls_dict[bb["cls"]] +"_real"), (bb["x1"], bb["y1"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    if color ==2:
        for bb in bboxes:
            cv2.rectangle(img, (bb["x1"], bb["y1"]), (bb["x2"], bb["y2"]), (0,255,255), 1)
            cv2.putText(img, str(
                idx_cls_dict[bb["cls"]] + "_detect"), (bb["x1"], bb["y1"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, cv2.LINE_AA)

    return img

def label_to_box(file_path, im_height, im_width):
    labels = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        image = cv2.imread(file_path.replace(".txt", ".jpg"))
        data = f.read().strip().splitlines()
        for dd in data:
            ll = {}

            cls, xc, yc, w, h = dd.split(" ")
            cls = int(cls)
            x1 = int(im_width * (float(xc) - float(w) / 2))
            y1 = int(im_height * (float(yc) - float(h) / 2))
            x2 = int(im_width * (float(xc) + float(w) / 2))
            y2 = int(im_height * (float(yc) + float(h) / 2))

            ll.update({"cls": int(cls)})
            ll.update({"x1": int(im_width * (float(xc) - float(w) / 2))})
            ll.update({"y1": int(im_height * (float(yc) - float(h) / 2))})
            ll.update({"x2": int(im_width * (float(xc) + float(w) / 2))})
            ll.update({"y2": int(im_height * (float(yc) + float(h) / 2))})
            labels.append(ll)
    return labels

def m_convert_box(detected_objects):
    return_boxes = []
    for d_o in detected_objects:
        _, label, (left_x, top_y, width, height), confidence, cam_id = d_o
        results = {}
        results.update({"cls": cls_dict[label]})
        results.update({"x1": int(left_x)})
        results.update({"y1": int(top_y)})
        results.update({"x2": int(left_x + width)})
        results.update({"y2": int(top_y + height)})
        return_boxes.append(results)

    return return_boxes

def cal_confusion_matrix(predict, labels):
    confusion_matrix = [[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]

    matrix = []
    for J in range(len(predict)):
        sub_mat = []
        for I in range(len(labels)):
            sub_mat.append(0.0)
        matrix.append(sub_mat)

    is_zero = []
    for I in range(len(labels)):
        is_zero.append(True)

    for J, bbox2 in enumerate(labels):
        for I, bbox1 in enumerate(predict):
            iou = get_iou(bbox1, bbox2)
            if iou > 0.8:
                matrix[I][J] = iou

    for J in range(len(predict)):
        max_value = 0
        max_index = -1
        for I in range(len(labels)):
            temp = matrix[J][I]
            if temp > 0:
                if (max_value < temp):
                    max_value = temp
                    max_index = I
        if max_index >= 0:
            confusion_matrix[predict[J]["cls"]][labels[max_index]["cls"]] += 1
            is_zero[max_index] = False

    for I in range(len(labels) - 1):
        if is_zero[I]:
            confusion_matrix[7][labels[I]["cls"]] += 1

    return np.array(confusion_matrix)


if __name__ == "__main__":
    yolov4 = YoloV4(darknet="/home/m/Desktop/LONG/data_nobi/darknet/darknet",
                    config_file="/home/m/Desktop/LONG/data_nobi/scaled_nobi_pose.cfg",
                    data_file="/home/m/Desktop/LONG/data_nobi/coco.data",
                    weights_file="/home/m/Desktop/LONG/data_nobi/scaled_nobi_pose_290000.weights")

    confusion_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0]])

    for imfile in tqdm(glob(os.path.join("/home/m/Desktop/LONG/data_nobi/M/*.jpg"))):
        image = cv2.imread(imfile)
        labels = label_to_box(imfile.replace(
            ".jpg", ".txt"), image.shape[0], image.shape[1])
        detected_objects = yolov4.process([image])
        # import ipdb;ipdb.set_trace()
        detected_objects = m_convert_box(detected_objects)

        draw_img = draw(image, labels, 1)
        draw_img = draw(image, detected_objects, 2)
        cv2.imwrite("/home/m/Desktop/LONG/output/" +
                    str(os.path.basename(imfile)), draw_img)

        confusion_matrix = confusion_matrix + \
            cal_confusion_matrix(detected_objects, labels)

    print(confusion_matrix)
