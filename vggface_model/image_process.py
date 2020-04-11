import cv2
import numpy as np
import os


def resize_images(images, new_dim):
    # images should be a list or other iterable objects
    height, width = new_dim[0], new_dim[1]
    dsts = []
    for img in images:
        dst = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
        dsts.append(dst)
    return dsts

def resize_image(img, new_dim):
    # print(img.shape)
    height, width = new_dim[0], new_dim[1]
    dst = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
    return dst

def rotate_image(img, degree):
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst