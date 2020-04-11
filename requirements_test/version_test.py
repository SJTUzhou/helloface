import tensorflow as tf
import cv2
import dlib
import tkinter as tk
import keras_vggface
import PIL
import cv2
import dlib
import numpy as np
import csv
import imutils
import matplotlib

'''
print("tensorflow version: {}".format(tf.__version__))
print("opencv version: {}".format(cv2.__version__))
print("dlib version: {}".format(dlib.__version__))
print("PIL version: {}".format(PIL.__version__))
print("numpy version :{}".format(np.__version__))
print("imutils version: {}".format(imutils.__version__))
print("keras_vggface version: {}".format(keras_vggface.__version__))
'''

# an easy way to install dlib of a old version  for python 3.6 on windows
# pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl
# a way to install keras_vggface
# pip install git+https://github.com/rcmalli/keras-vggface.git
# a way to install module imutils
# pip install imutils

requires = ["python version: 3.6.9", "tensorflow version: {}".format(tf.__version__),\
    "opencv version: {}".format(cv2.__version__), "dlib version: {}".format(dlib.__version__),\
    "PIL version: {}".format(PIL.__version__), "numpy version :{}".format(np.__version__),\
    "keras_vggface version: {}".format(keras_vggface.__version__),\
    "matplotlib version: {}".format(matplotlib.__version__)]
with open("../requirements.txt", 'w') as f:
    for require in requires:
        print(require)
        f.write(require+'\n')