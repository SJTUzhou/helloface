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
import image_collect.path_const as const
from vggface_model.vggface_utils import VggfaceUtils
from vggface_model.face_detection import DetectionUtils

def video_test():
    detectionUtils = DetectionUtils()
    vggfaceUtils = VggfaceUtils()
    names, centers = vggfaceUtils.read_name_center()
    camera = cv2.VideoCapture(0)
    while True:
        _, img = camera.read()
        face_positions = detectionUtils.get_face_positions(img, use_dlib=True)
        faces = detectionUtils.get_face_regions(face_positions, img)
        predictions = vggfaceUtils.model_predict(faces)
        pred_names = vggfaceUtils.predict_names(predictions)
        detectionUtils.draw_faces_and_names(face_positions, pred_names, img)
        cv2.imshow('Video', img)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_test()
