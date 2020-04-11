import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import cv2
import numpy as np
import dlib


class DetectionUtils(object):
    def __init__(self):
        self.dlib_detector = dlib.get_frontal_face_detector()
        haarcascade_file = "../info/haarcascade_frontalface_default.xml"
        self.haarcascade_detector = cv2.CascadeClassifier(haarcascade_file)
    
    def get_face_rects(self, img):
        """ This function is only used by dlib module !!! to get the face Rectangles
        Parameters:
            img (numpy.ndarray)
        Return:
            A list of dlib.Rectangles describing the faces
        """ 
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.dlib_detector(gray_img, 0)
        return rects

    def get_face_positions(self, img, use_dlib):
        """ get the face position on an image
        Parameters:
            img (numpy.ndarray)
            use_dlib (bool) : whether to use module dlib to detect faces
        Return:
            None or A list of tuples describing face positions (left, top, width, height)
        """
        positions = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if use_dlib:
            rects = self.dlib_detector(gray_img, 0)
            if rects is None:
                return None
            else:
                for face in rects:
                    y1 = face.top() if face.top() > 0 else 0
                    y2 = face.bottom() if face.bottom() > 0 else 0
                    x1 = face.left() if face.left() > 0 else 0
                    x2 = face.right() if face.right() > 0 else 0
                    positions.append((x1, y1, x2-x1, y2-y1))
                return positions
        else:
            dets = self.haarcascade_detector.detectMultiScale(gray, 1.1, 4)
            return list(dets)
    
    def get_face_regions(self, face_positions, image):
        """ crop the face region on an image
        Parameters:
            face_positions (list) : A list of tuples shaped like (x, y, w, h)
            image (numpy.ndarray) : crop source image
        Return:
            faces (list) : A list of face images (numpy.ndarray)
        """
        faces = []
        for (x, y, w, h) in face_positions:
            face = image[y:y+h, x:x+w, :]
            faces.append(face)
        return faces

    def draw_faces_on_image(self, face_positions, image):
        for (x, y, w, h) in face_positions:
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), thickness=2)
        
    def draw_name_on_image(self, face_positions, names, image):
        for position, name in zip(face_positions, names):
            x, y, w, h = position
            cv2.putText(image, name, (x,y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(255,0,0), thickness=2) 

    def draw_faces_and_names(self, face_positions, names, image):
        for position, name in zip(face_positions, names):
            x, y, w, h = position
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), thickness=2)
            cv2.putText(image, name, (x,y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(255,0,0), thickness=2) 
    




