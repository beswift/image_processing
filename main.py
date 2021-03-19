import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import glob
from feature_detectors import *
from Image_processors import *
import pytesseract as ts
from pytesseract import Output
from pdf2image import convert_from_path, convert_from_bytes
from io import BytesIO

# Sidebar Section
st.sidebar.write('''
# Image processing playground
''')
st.sidebar.subheader("Pick an area to explore")
playground = st.sidebar.radio("Playgrounds", ['Image Alignment','Image Processing','OCR'])

# header section
header = st.empty()

img_types = ['image/jpg', 'image/png', 'image/jpeg', 'image/tiff']
doc_types = ['application/pdf']
ata_types = ['csv']

def file_uploader(function_name):
    img_buff = st.file_uploader("Upload an image to {}".format(function_name))
    return img_buff

# Image Alignment
if playground == 'Image Alignment':
    header = st.header('Image Alignment')
    '''
    How do we register images so that they can be accurately aligned?
    
    [1] Convert input image to standard image object
    
    [2] Convert images to grayscale 
    
    [3] Enhance image to amplify curves
    
    [4] (fundus photos) Mask image to remove background
    
    [5] Feature detection to detect corners ( ** test if actually needed with SIFT/SURF)
    
    [6] Implement and Compare SURF and SIFT based feature detection
    
    [7] Create composite image using detected points from step 5
    
    '''

    toolbox = st.beta_container()

    with toolbox:
        func_col, space_col, result_col = st.beta_columns((2,.5,4))

    with func_col:
        if st.button('Test compare'):
            compare_images()

    with result_col:
        func_image = file_uploader("compare")


    working_folder = st.selectbox("pick a working folder",os.listdir(os.getcwd()))
    images_folder = os.path.join(os.getcwd(),working_folder).lower()
    for image in glob.glob(('{}//*.*'.format(images_folder))):

        st.image(image)
        image_path = os.path.join(images_folder, image)
        st.write(image_path)
        imageIn = cv2.imread(image_path)
        gray = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)
        # sift_features(imageIn,image)
        # surf_features(imageIn,image)
        # get_harris_corners(imageIn,image)
        # countour_mask(imageIn,image)

    # compare_images()

# Image Processing

if playground == 'Image Processing':
    header = st.header('Computer Vision for human vision')

    '''
    What kind of things need to be done to images to make them more usable?
    
    [ ] Single file multipage tiff merging (for Optos primary images)
    
    [ ] Image Enhancement  - normalize image and adjust brightness contrast
    
    [ ] Image Filters - RGB filter based on frequency for better layer imaging
    
    [ ] Raw OCT file parsing 
    '''
    proc_container = st.beta_container()

    with proc_container:
        description_col, space, img_col = st.beta_columns((2,.5,4))

    with description_col:
        st.write("merge tiffs..")

    with img_col:
        m_tiff = file_uploader("merge")
        



# OCR

if playground == 'OCR':
    header = st.header('Adventures in text recognition')

    '''
    ## How can we get patient info off of image reports?
    
     [1] Get an image  - upload an image file 
     
     [2] Display bounding boxes for recognized text
     
     [3] Assign values to bounding boxes
     
     [4] Store values in PHI array
    '''

    #

    img_file_buffer = file_uploader("read")

    image_overview_section = st.beta_container()
    with image_overview_section:
        img_col, space_col, ocr_col = st.beta_columns((4, 1, 4))

    if img_file_buffer is not None:
        st.write(img_file_buffer.type)
    with img_col:
        if img_file_buffer.type in doc_types:

            st.write("document detected, converting to image..")
            bytes_img = img_file_buffer.read()
            img_file_buffered = convert_from_bytes(bytes_img, timeout=8000)

            st.write(img_file_buffered)
            st.image(img_file_buffered, use_column_width=True)
            pre_img = convert2jpg(img_file_buffered[0])
            image = Image.open(BytesIO(pre_img))

        elif img_file_buffer.type in img_types:
            st.write("image detected..")
            image = Image.open(img_file_buffer)
            st.write(image)
            st.image(image, caption="The caption", use_column_width=True)
            # test_image = cv2.imread(img_array)

    with ocr_col:
        st.write(type(image))
        pre_img = np.array(image)  # for opencv
        st.write(type(pre_img))
        proc_image = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        d = ts.image_to_data(proc_image, output_type=Output.DICT)
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(pre_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.text(d['text'][i])
            st.write("{},{},{},{}".format(x, y, w, h))

    with img_col:
        d = ts.image_to_data(proc_image, output_type=Output.DICT)
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(pre_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        st.image(pre_img)
