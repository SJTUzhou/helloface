import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tkinter as tk
import tkinter.messagebox as messagebox
import keras_vggface
from PIL import Image, ImageTk
from vggface_model.vggface_utils import VggfaceUtils
from vggface_model.face_detection import DetectionUtils
from image_collect.get_my_photos import get_faces
import image_collect.path_const as const
import cv2
import dlib
import numpy as np
import csv
import time
from imutils import face_utils
from decoration.add_decoration import add_beard,add_glasses,add_hat
import random
from copy import deepcopy

class FaceRecogGUI(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent)
        # attributes for face detection and the camera
        self.camera = cv2.VideoCapture(0)
        self.vggfaceUtils = VggfaceUtils()
        self.detectionUtils = DetectionUtils()
        # read the names and their vector centers from the csv file
        self.names, self.centers = self.vggfaceUtils.read_name_center()
        # BGR color images to add decorations
        self.origin_beard = cv2.imread('../info/beard.jpg',1)
        self.origin_hat = cv2.imread('../info/hat.jpg',1)
        self.origin_glasses = cv2.imread('../info/glasses.jpg',1)
        self.shape_predictor = dlib.shape_predictor(const.SHAPE_PREDICTOR_FILE)# get the 68 points of the face.
        # a flag to control whether to add decoration or not
        self.flag_decor = False

        # show the image from the camera: left side of the main window
        imageFrame = tk.Frame(parent)
        imageFrame.grid(row=0, rowspan=5, column=0, padx=5, pady=10)
        self.panel = tk.Label(imageFrame)
        self.panel.grid(row=0, rowspan=5, column=0)

        # a frame to hold all the other widgets in the right side
        buttonFrame = tk.Frame(parent)
        buttonFrame.grid(row=0, rowspan=3, column=1, padx=5, pady=10)
        # a label to indicate the text entry following it
        var_new_name = tk.StringVar()
        var_new_name.set('Enter New Name')
        label_new_name = tk.Label(buttonFrame, textvariable=var_new_name, width=20, font=('Arial', 16))
        label_new_name.grid(row=0, column=1, pady=5)
        # a text entry to input a new name to the app
        self.new_name = tk.StringVar()
        entry_new_name = tk.Entry(buttonFrame, textvariable=self.new_name, font=('Arial', 16))
        entry_new_name.grid(row=1, column=1, pady=15)
        # 3 buttons to control the functions
        btFontSize = 14
        # function 1: add a new person's name to the database and detect him or her right now after the database integration
        bt_new_name = tk.Button(buttonFrame, text='Add Name', height=3, width=20, command=self.take_photos_and_load_data, font=('Arial', btFontSize))
        bt_new_name.grid(row=2, column=1, pady=10)
        # function 2: add the interesting decorations including hat, beard and sunglasses
        bt_add_decoration = tk.Button(buttonFrame, text='Decorate or Not', height=3, width=20, font=('Arial', btFontSize), command=self.add_or_remove_decoration)
        bt_add_decoration.grid(row=3, column=1, pady=10)

        # function 3: check all the strangers and write down their images in a folder
        self.intvar_check = tk.IntVar()
        self.stranger_count = 0
        bt_check_stranger = tk.Checkbutton(buttonFrame, text='Record Stranger', height=3, width=20, font=('Arial', 12), variable=self.intvar_check,\
            onvalue=1, offvalue=0)
        bt_check_stranger.grid(row=4, column=1, pady=5)
        self.update()

    def take_photos_and_load_data(self):
        # take photos of the new person after clicking the Add Name button
        name = self.new_name.get()
        if len(name)>0:
            # get the faces and show them out in another small window
            get_faces(const.train_data_path, name, max_num=100, camera=self.camera)
            self.new_name.set('')
            # load the new name and its vector to the csv file
            self.vggfaceUtils.load_new_name_center(name)
            self.names, self.centers = self.vggfaceUtils.read_name_center()
        else:
            messagebox.showwarning(title="Invalid", message="An Empty Name!")

    def update(self):
        # show the image output by the face detection model
        # this function is always working during the life cycle of the main window
        success, img = self.camera.read()
        img = cv2.flip(img, 1)
        face_positions = self.detectionUtils.get_face_positions(img, use_dlib=True)
        if len(face_positions)>0:
            faces = self.detectionUtils.get_face_regions(face_positions, img)
            predictions = self.vggfaceUtils.model_predict(faces)
            pred_names = self.vggfaceUtils.predict_names(predictions)
            self.detectionUtils.draw_faces_and_names(face_positions, pred_names, img)
            if self.intvar_check.get() == 1:
                if  self.vggfaceUtils.stranger_label in pred_names:
                    self.stranger_count += 1
                    if self.stranger_count > 5:
                        localtime = time.asctime(time.localtime(time.time()))
                        file_name = 'stranger'+'-'+str(localtime).replace(' ', '_').replace(":", "-")+'.jpg'
                        save_file_path = os.path.join(const.STRANGER_SAVE_PATH, file_name)
                        cv2.imwrite(save_file_path, img)
                        self.stranger_count = 0
                else:
                    self.stranger_count = 0

            if self.flag_decor:
                face_rects = self.detectionUtils.get_face_rects(img)
                for rect in face_rects:
                    self.add_decoration(rect, img)
        # transform the form of the image to display it
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        # call the function after every 10 ms
        self.panel.after(10, lambda:self.update())
    
    def add_decoration(self, face_rect, image):
        # Make the prediction and transfom it to numpy array
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = self.shape_predictor(gray, face_rect)
        shape = face_utils.shape_to_np(shape)
        # Draw on our image, all the finded cordinate points (x,y) 
        add_beard(self.origin_beard,image,shape)
        add_hat(self.origin_hat,image,shape)
        add_glasses(self.origin_glasses,image,shape)

    def add_or_remove_decoration(self):
        self.flag_decor = not self.flag_decor

    def release(self):
        self.camera.release()
        cv2.destroyAllWindows()

    '''
    def detect_save_stranger(self, save_stranger=const.STRANGER_SAVE_PATH,img_size=224, max_num=50):
        detector = dlib.get_frontal_face_detector()
        camera=self.camera
        if camera is None:
            camera = cv2.VideoCapture(0)

        name_center_csv_file = NAME_CENTER_CSV
        names, centers = read_name_center(name_center_csv_file)

        index = 0
        count = 0
        global sign
        sign = 0
        while count<=max_num:
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
                # get the prediction vector
                prediction = characterize_one_face(face, self.model)
                # get the predict name
                name = predict_name(prediction, names, centers)
              
            if name not in names:
                sign += 1
                if sign>=10:   
                    face = cv2.resize(face, (img_size,img_size))
                    cv2.imshow("Face", face)
                    cv2.imwrite(os.path.join(save_stranger,'stranger'+'-'+str(index)+'.jpg'), face)
                    index += 1
                    print('Being processed picture %s / %s' % (index, max_num))
                    sign = 0
            else:
                sign=0
            count += 1
            if cv2.waitKey(30)&0xff == 27:
                break
        if camera is None:
            camera.release()
        cv2.destroyWindow("Face")
        print("Finish!")
    '''

if __name__ == "__main__":
    main_window = tk.Tk()
    main_window.title("Face Recognition")
    app = FaceRecogGUI(parent=main_window)
    app.mainloop()
    app.release()
