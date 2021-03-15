import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import glob
from feature_detectors import *
from Image_processors import *
import pytesseract as ts
from pdf2image import convert_from_path, convert_from_bytes

# Sidebar Section
st.sidebar.write('''
# Image processing playground
''')
st.sidebar.subheader("Pick an area to explore")
playground = st.sidebar.radio("Playgrounds", ['Image Alignment', 'OCR'])

# header section
header = st.empty()

# Image Alignment
if playground == 'Image Alignment':
    header = st.header('Image Alignment')

    if st.button('Test compare'):
        compare_images()

    images_folder = os.path.join(os.getcwd(), "test_images")
    st.write(images_folder)

    for image in glob.glob(('{}//*.jpg'.format(images_folder))):
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

    img_file_buffer = st.file_uploader("Upload an image")
    img_types = ['image/jpg', 'image/png', 'image/jpeg', 'image/tiff']
    doc_types = ['application/pdf']
    data_types = ['csv']

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
                pre_img = img_file_buffered

        elif img_file_buffer.type in img_types:
            st.write("image detected..")
            image = Image.open(img_file_buffer)
            st.write(image)
            st.image(image, caption="The caption", use_column_width=True)
            # test_image = cv2.imread(img_array)
            pre_img = np.array(image)  # for opencv
    with ocr_col:
        proc_image = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        image_text = ts.image_to_string(proc_image)
        st.text(image_text)
