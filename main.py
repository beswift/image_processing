#import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import glob
from feature_detectors import *
from Image_processors import *



images_folder = os.path.join(os.getcwd(), "test_images")
print(images_folder)



for image in glob.glob(('{}//*.jpg'.format(images_folder))):
    print(image)
    image_path =os.path.join(images_folder,image)
    print (image_path)
    imageIn = cv2.imread(image_path)
    gray = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)
    # sift_features(imageIn,image)
    # surf_features(imageIn,image)
    # get_harris_corners(imageIn,image)
    countour_mask(imageIn,image)


compare_images()









