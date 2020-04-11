import keras_vggface
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import cv2
from vggface_model.image_process import resize_image, resize_images
import image_collect.path_const as const
import numpy as np
import csv

class VggfaceUtils(object):
    def __init__(self):
        # build a vggface model to get the prediction vector
        self.img_shape = (224, 224, 3)
        self.model = keras_vggface.VGGFace(model='resnet50', include_top=False, input_shape=self.img_shape, pooling='avg')
        self.csv_file = const.NAME_CENTER_CSV
        self.traindata_path = const.train_data_path
        self.stranger_label = "stranger"
    
    def load_images(self, dir_path):
        """ load all the images in a directory
        Parameters:
            dir_path (str) : Directory where to load the images
        Return:
            images (list) : A list of images (numpy.ndarray)
        """ 
        images = []
        for image_file in os.listdir(dir_path):
            images.append(cv2.imread(os.path.join(dir_path, image_file)))
        return images

    def preprocess(self, imgs):
        """ use image operations to fit the images to the input form
        Parameters:
            imgs (list) : A list of images (numpy.ndarray)
        Return:
            input_imgs (numpy.ndarray) : an array of images with its shape (None, 224, 224, 3)
        """ 
        input_imgs = resize_images(imgs, self.img_shape[:2])
        input_imgs = np.array(input_imgs)
        return input_imgs
    
    def model_predict(self, imgs):
        # imgs actually are the detected faces
        if len(imgs)==0:
            return None
        else:
            input_imgs = self.preprocess(imgs)
            predictions = self.model.predict(input_imgs)
            # predictions vector shape (None, 2048)
            return predictions
    
    def get_predictions_center(self, predictions):
        # predictions is a numpy ndarray with its shape of (None, 2048)
        mean_center = np.mean(predictions, axis=0)
        # mean center shape (2048,)
        return mean_center

    def get_images_center_in_dir(self, dir_path):
        # images is a numpy ndarray of images with different sizes
        images = self.load_images(dir_path)
        predictions = self.model_predict(images)
        # get the average vector of the images of the same person
        # mean center shape (2048,)
        center = self.get_predictions_center(predictions)
        return center
    
    def write_images_center(self, names, centers):
        # write the face image vector centers into a csv file
        with open(self.csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            center_dim = len(centers[0])
            # create the csv header
            header = ['name',]
            header.extend(["x%d"%(i) for i in range(0,center_dim)])
            writer.writerow(header)
            # write the content by row
            for name, center in zip(names, centers):
                info = [name,]
                info.extend([x for x in center])
                writer.writerow(info)
    
    def read_name_center(self):
        # read names and their face vectors' center from the csv file
        # output a list of names and a np array of centers
        names = []
        centers = []
        with open(self.csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                # print(row)
                center = [float(elem) for elem in row[1:]]
                centers.append(center)
                names.append(row[0])
        return names, np.array(centers)

    def load_all_train_data_centers(self):
        # calculate the average vector center for each name in the traindata directory
        # write them down in the csv file
        names = os.listdir(self.traindata_path)
        centers = []
        image_dirs = [os.path.join(self.traindata_path, name) for name in names]
        for image_path in image_dirs:
            # use an existing model to predict the cluster center
            center = self.get_images_center_in_dir(image_path)
            centers.append(center)
            print("Finish loading {} vector center.".format(image_path))
        self.write_images_center(names, centers)
        print("Finish loading")

    def load_new_name_center(self, new_name):
        center = self.get_images_center_in_dir(os.path.join(self.traindata_path, new_name))
        with open(self.csv_file, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            info = [new_name,]
            info.extend([x for x in center])
            writer.writerow(info)
    
    def predict_names(self, predictions, threshold=100):
        # return a list of predicted names
        pred_names = []
        names, centers = self.read_name_center()
        repeat_predictions = np.repeat(predictions[:,np.newaxis,:], repeats=len(names), axis=1)
        repeat_centers = np.repeat(centers[np.newaxis,:,:], repeats=predictions.shape[0], axis=0)
        delta = repeat_centers - repeat_predictions
        # delta shape (predictions_num, names_num, 2048)
        all_distances = np.sqrt(np.sum(np.square(delta),axis=2))
        # set a threshold: A distance larger than the threshold will be regarded as stranger's vector.
        for distances in list(all_distances):
            if np.min(distances)>threshold:
                pred_names.append(self.stranger_label)
            else:
                min_idx = np.argmin(distances)
                pred_name = names[min_idx]
                pred_names.append(pred_name)
        return pred_names





