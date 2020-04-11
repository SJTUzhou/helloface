import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from imutils import face_utils
import dlib
import cv2
import numpy as np
from image_collect.path_const import *

'''
This is a module used for adding decoration to the pictures
It includes three funcitions add_beard,add_glasses,and add_hat
The three funcitons are based on the landmarks on this detected face
By file "shape_predictor_68_face_landmarks.dat" in the folder info
We can have a list of 68 characteristic points(tuple)
    IdxRange jaw --> [0,16]
    IdxRange rightBrow --> [17,21]
    IdxRange leftBrow --> [22,26]
    IdxRange nose --> [27,35]
    IdxRange rightEye --> [36,41]
    IdxRange leftEye --> [42,47]
    IdxRange mouth --> [48,59]
    IdxRange mouth_inside --> [60,67]
'''

def add_beard(origin_beard,image,shape):
    '''
    This funciton is used for adding beard to the picture
    :param origin_beard: the image of beard you want to add to the picture
    :type origin_beard:numpy.ndarray
    :param image: the picture that you want to add beard to
    :type image: numpy.ndarray
    :param shape:a list of 68 characteristic points(tuple) --> landmarks on this detected face
    :type shape: numpy.ndarray from list of tuple
    It uses the place of mouth and nose to find the place of beard
    The width of beard is equal to that of mouth
    The beard is between the nose and the mouth
    ''' 
    mouth = np.array(shape[48:55])
    nose = np.array(shape[27:35])
    # the width of beard is equal to that of mouth
    left = np.min(mouth[:,0])
    right = np.max(mouth[:,0])
    # place the beard between the nose and the mouth
    up = np.max(nose[:,1])
    down = np.min(mouth[:,1])
    # avoid the 'out of index' error
    if (right-left)>0 and (down-up)>0 and up>0 and left>0 and right<image.shape[1]:
        beard = cv2.resize(origin_beard, (right-left,down-up))        
        thre_color = 50
        #mask is a bool ndarray to eliminate the grounding
        mask = beard<thre_color
        mouth = image[up:down, left:right, :]
        mouth[mask] = beard[mask]


def add_hat(origin_hat,image,shape):
    '''
    This funciton is used for adding hat to the picture
    :param origin_hat: the image of hat you want to add to the picture
    :type origin_hat:numpy.ndarray
    :param image: the picture that you want to add hat to
    :type image: numpy.ndarray
    :param shape:a list of 68 characteristic points(tuple) --> landmarks on this detected face
    :type shape:numpy.ndarray form list of tuple
    It uses the place of brows and jaw to find the place of hat
    The width of hat is equal the width of jaw times 1.2
    The place of the hat is that the place of the top of jaw moves up the distance between the bottom of brow and the bottom of jaw plus 10
    '''
    brows = np.array(shape[17:27])    
    jaw = np.array(shape[0:17])
    down_brow= np.min(brows[:,1])
    left = np.min(jaw[:,0])
    right = np.max(jaw[:,0])
    up = np.min(jaw[:,1])
    down = np.max(jaw[:,1])
    #using the distance between the brow and jaw instead of directly a number to make it more accurate for different faces
    dis = down-down_brow+10
    #The width of hat is equal the width of jaw times 1.2
    #The place of the hat is that the place of the top of jaw moves up the distance between the bottom of brow and the bottom of jaw plus 10
    hat = cv2.resize(origin_hat, (int(1.2*(right-left)),down-up))        
    #force the move to be int and avoid the "out of index"
    left_move = (right - left)//10
    right_move = int(0.2*(right-left))-left_move
    # avoid the 'out of index' error
    if (up-dis)>0 and (left-left_move)>0 and (right+right_move)<image.shape[1]:
        #mask is a bool ndarray to eliminate the grounding
        mask = hat<150    
        region_hat = image[up-dis:down-dis, left-left_move:right+right_move, :]
        region_hat[mask] = hat[mask]


def add_glasses(origin_glasses,image,shape):
    '''
    This funciton is used for adding glasses to the picture
    :param origin_glasses: the image of glasses you want to add to the picture
    :type origin_glasses:numpy.ndarray
    :param image: the picture that you want to add glasses to
    :type image: numpy.ndarray
    :param shape:a list of 68 characteristic points(tuple) --> landmarks on this detected face
    :type shape:numpy.ndarray from list of tuple
    It uses the place of brows,eyes and jaw to find the place of glasses
     
    '''
    brows = np.array(shape[17:27])
    eyes = np.array(shape[36:48])
    jaw = np.array(shape[0:17])
    up= np.max(brows[:,1])
    left = np.min(jaw[:,0])
    right = np.max(jaw[:,0])
    down = np.max(eyes[:,1])
    # avoid the 'out of index' error
    if up>0 and (left+10)>0 and (right-10)<image.shape[1]:
        #The width of glasses is equal to that of mouth minus 20
        #The glasses is between the brows and the eyes and the height is equal to the distance betwwen the brows and the eyes times 2.4
        glasses = cv2.resize(origin_glasses,(right-left-20,int(2.4*(down-up))))        
        thre_color = 10
        up_move= int(0.4*(down-up))
        down_move= int(1.4*(down-up))-up_move
        #mask is a bool ndarray to eliminate the grounding
        mask = glasses<200
        region_glasses = image[up-up_move:down+down_move, left+10:right-10, :]
        region_glasses[mask] = glasses[mask]

    
def add_pictures_test():
    # landmarks on this detected face
    # p = our pre-treined model directory, on my case, it is in the info directory.
    p = SHAPE_PREDICTOR_FILE
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    
    # BGR color image beard
    origin_beard = cv2.imread('../info/beard.jpg',1)
    origin_hat = cv2.imread('../info/hat.jpg',1)
    origin_glasses = cv2.imread('../info/glasses.jpg',1)
    
    cap = cv2.VideoCapture(0)

    while True:
        # Getting out image by webcam 
        _, image = cap.read()
        # Converting the image to gray scale
        image = cv2.flip(image, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get faces into webcam's image
        rects = detector(gray, 0)
    
        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            #Draw on our image, all the finded cordinate points (x,y) 
            add_beard(origin_beard,image,shape)
            add_hat(origin_hat,image,shape)
            add_glasses(origin_glasses,image,shape)

        # Show the image
        cv2.imshow("Output", image)
    
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__=="__main__":
    add_pictures_test()