import cv2
import dlib
import os
import sys
import random
import numpy as np
from image_collect.path_const import *

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst


def get_faces(path, name, max_num, img_size=224, camera=None):
    save_dir = os.path.join(path, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(name_path, 'a+') as f:
        f.seek(0)
        names = f.read().split(',')
        if name not in names:
            f.write(name+',')
    # use frontal_face_detector of dlib as the feature extractor
    detector = dlib.get_frontal_face_detector()
    if camera is None:
        camera = cv2.VideoCapture(0)
    index = 1
    while index<=max_num:
        print('Being processed picture %s / %s' % (index, max_num))
        # read photos from camera
        success, img = camera.read()
        img = cv2.flip(img,1,dst=None)  
        # convert to grayscale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # use detector for facerecogniton
        dets = detector(gray_img, 1)
        for d in dets:
            y1 = d.top() if d.top() > 0 else 0
            y2 = d.bottom() if d.bottom() > 0 else 0
            x1 = d.left() if d.left() > 0 else 0
            x2 = d.right() if d.right() > 0 else 0
            face = img[y1:y2,x1:x2]
            # adjust the contrast and brightnenss of the photo, the values of the contrast and brightness are random which can increase samples diversity
            face = Contrast_and_Brightness(random.uniform(0.8, 1.2), random.randint(-20, 10),face)
            face = cv2.resize(face, (img_size,img_size))
            cv2.imshow("Face", face)
            cv2.imwrite(os.path.join(save_dir, name+'-'+str(index)+'.jpg'), face)
            index += 1
        if cv2.waitKey(30)&0xff == 27:
            break
    if camera is None:
        camera.release()
    cv2.destroyWindow("Face")
    print("Finish!")
            

def photo_taker_loop(path,max_num): 
    name = input("Please enter the new name: ")
    while True:
        action = input('''Enter 'c' to change the name\nEnter 'p' to start get the phote\nEnter 'q' to break\nYour choice: ''')
        if action=='c':
            name = input("Please enter the new name: ")
        if action=='p':
            get_faces(path, name, max_num, img_size=224)
        if action=='q':
            break

if __name__ == "__main__":
    #photo_taker_loop(train_data_path,200)
    photo_taker_loop(test_data_path,50)
    