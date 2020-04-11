import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import keras_vggface
import cv2
import dlib
import numpy as np
import csv
from vggface_model.vggface_utils import VggfaceUtils
from vggface_model.face_detection import DetectionUtils
import image_collect.path_const as const
from image_collect.img_rename import rename_image

'''
This is a module used for doing accuracy test.
We test the recognition accuracy with 500 pictures.(200 of known students and 300 of strangers)
This module is also used in optimizing parameter function

'''


def accuracy(vggfaceUtils, detectionUtils, threshold):

    names,centers = vggfaceUtils.read_name_center() 

    undetected_num = 0

    faces = []
    labels = []
    def crop_face(img_file):
        img = cv2.imread(img_file)
        face_positions = detectionUtils.get_face_positions(img, True)
        if len(face_positions) > 0:
            faces = detectionUtils.get_face_regions(face_positions, img)
            return faces[0]
        else:
            return None

    path = const.test_data_path
    rename_image(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            image_file = os.path.join(root, file)
            face = crop_face(image_file)
            if face is not None:
                labels.append(file.split('-')[0])
                faces.append(face)
            else:
                undetected_num += 1

    path = const.stranger_data_path
    rename_image(path)
    for file in os.listdir(path):
        image_file = os.path.join(path, file)
        face = crop_face(image_file)
        if face is not None:
            labels.append(file.split('-')[0])
            faces.append(face)
        else:
            undetected_num += 1

    predictions = vggfaceUtils.model_predict(faces)
    pred_names = vggfaceUtils.predict_names(predictions, threshold=threshold)

    accurate_num = 0
    total_num = len(labels) + undetected_num
    for name,label in zip(pred_names, labels):
        if name == label:
            accurate_num += 1
    accurate_rate = accurate_num/total_num
    print("accurate rate is: {}".format(accurate_rate))
    return accurate_rate


if __name__ == "__main__":
    vggfaceUtils = VggfaceUtils()
    detectionUtils = DetectionUtils()
    accuracy(vggfaceUtils, detectionUtils, threshold=100)



