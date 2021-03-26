import io

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import glob

from tifffile import TiffFile

from feature_detectors import *
from Image_processors import *
import pytesseract as ts
from pytesseract import Output
from pdf2image import convert_from_path, convert_from_bytes
from io import BytesIO
from PIL import Image
import tifffile
from skimage import exposure

# Sidebar Section
st.sidebar.write('''
# Image processing playground
''')
st.sidebar.subheader("Pick an area to explore")
playground = st.sidebar.radio("Playgrounds", ['Image Alignment', 'Image Processing', 'OCR'])

# header section
header = st.empty()

img_types = ['image/jpg', 'image/png', 'image/jpeg', 'image/tiff']
doc_types = ['application/pdf']
ata_types = ['csv']


def file_uploader(function_name):
    img_buff = st.file_uploader("Upload an image to {}".format(function_name))
    return img_buff


def files_uploader(function_name):
    img_buff = st.file_uploader("Upload an image to {}".format(function_name), accept_multiple_files=True)
    return img_buff[0]


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
        func_col, space_col, result_col = st.beta_columns((2, .5, 4))

    with func_col:
        if st.button('Test compare'):
            compare_images()

    with result_col:
        '''
        ### upload here
        '''
        # TODO: use files_uploader here to pass list to compare function
        # func_image = file_uploader("compare")

    '''
    ## Test running sift on a folder of matching images..
    '''
    #mask_Xcenter = st.slider("mask x center", 0,1980,800,1)
    #mask_Ycenter = st.slider("mask y center",0,1980,800,1)
    #mask_radius = st.slider("mask radius",0,2000,1000,1)
    Image.MAX_IMAGE_PIXELS = 933120000
    working_folder = st.selectbox("pick a working folder", os.listdir(os.getcwd()))
    images_folder = os.path.join(os.getcwd(), working_folder).lower()
    groupies = []
    for image in glob.glob(('{}//*.*'.format(images_folder))):
        groupies = groupies
        image_path = os.path.join(images_folder, image)
        st.write(image_path)
        imageIn = cv2.imread(image_path)
        sift_image = sift_features(imageIn)
        input_img = pre_stitch(imageIn)
        groupies.append(input_img)
        st.image(sift_image,caption=image)

    st.write(groupies)
    stiched = stitch_images(groupies)

    st.image(stiched[1])


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
    '''
    
    # 
    
    '''

    '''
    ## Process Tiff files:


    '''
    info_container = st.beta_container()

    with info_container:
        description_col, space, img_col = st.beta_columns((2, .5, 4))


    with description_col:
        tiff_folder = os.path.join(os.getcwd(),"optos_tiff")

        '''
        ###  First, pick a Tiff file to explore on the right
        
        '''
    with img_col:
        # props
        tiff_types = ["tif","tiff","TIF","TIFF"]
        destination = None
        upload_folder = "./_working/uploaded_files/"

        # upload a file
        file_object = file_uploader("merge")
        tiff_name = file_object.name
        if file_object is not None:
            #image_file = generate_tiff_files(file_object)
            #bytes_img = file_object.read()
            img_file_buffered = imread(file_object)

        #working_file = cv2.imwrite(os.path.join(upload_folder,tiff_name),img_file_buffered)
        #m_tiff = img_file_buffered
        m_tiff = os.path.join(tiff_folder,tiff_name)


    basic_container = st.beta_container()

    with basic_container:
        description_col, space,img_col = st.beta_columns((2,.5,4))

    with description_col:
        ### Set options here

        brightness = st.slider("brightness:", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
        contrast = st.slider("contrast:",min_value=-10.0 , max_value=10.0 , value=1.0, step=0.1 )

    with img_col:
        # get file layers
        blue, green, red = create_tiff_layers(m_tiff)

        # Basic Comparison
        brg_compare = np.concatenate((blue, green, red), axis=1)

        # Merge Layers - rgb
        brg_merged = cv2.merge([red, green, blue])

        # Display Images

        bgr_display = st.image(brg_compare, caption="Layer Comparison")

        st.image(brg_merged, caption='output image')
        save_base_img = st.button("save output image")
        if save_base_img:
            cv2.imwrite("./optos_tiff/{}_base-merged.png".format(file_object.name),
                        cv2.cvtColor(brg_merged, cv2.COLOR_BGR2RGB))

    scale_container = st.beta_container()

    with scale_container:
        description_col, space, img_col = st.beta_columns((2, .5, 4))

    with description_col:

        alpha = st.slider("select an Alpha Value:",min_value=0.0, max_value= 510.0, value= 300.0, step=.5)
        beta = st.slider("select a Beta Value:", min_value=0.0, max_value=255.0, value=0.0,step =.5)

    with img_col:
        # Scale image channel

        blue_scaled = cv2.normalize(blue, dst=destination, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
        green_scaled = cv2.normalize(green, dst=destination, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
        red_scaled = cv2.normalize(red, dst=destination, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)

        scaled_compare = np.concatenate((red_scaled, green_scaled, blue_scaled), axis=1)

        scaled_merge = cv2.merge([red_scaled, green_scaled, blue_scaled])

        st.image(scaled_compare, caption="Scaled Comparison")
        st.image(scaled_merge, caption="scaled_output")
        save_scaled_img = st.button("save scaled image")
        if save_base_img:
            cv2.imwrite("./optos_tiff/{}_scaled-merged.png".format(file_object.name),
                        cv2.cvtColor(brg_merged, cv2.COLOR_BGR2RGB))

    clahe_container = st.beta_container()

    with clahe_container:
        description_col, space, img_col = st.beta_columns((2, .5, 4))

    with description_col:

        cliplimit = st.slider("select a clip limit", min_value=0.0, max_value = 100.0, value = 1.9, step=.1)
        tileGridSize = st.select_slider("select a tile grid size", options=[(1,1),(2,2),(3,3),(5,5),(8,8),(13,13),(21,21)])

    with img_col:

        # CHLAE image channels
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tileGridSize)

        red_clahe = clahe.apply(red)
        blue_clahe = clahe.apply(blue)
        green_clahe = clahe.apply(green)

        clahe_compare = np.concatenate((red_clahe, green_clahe, blue_clahe), axis=1)

        clahe_merged = cv2.merge([red_clahe, green_clahe, blue_clahe])


        st.image(clahe_compare, caption="clahe comparison")
        st.image(clahe_merged, caption="clahe output")
        save_clahe_img = st.button("save clahe image")
        if save_base_img:
            cv2.imwrite("./optos_tiff/{}_clahe-merged.png".format(file_object.name),
                        cv2.cvtColor(brg_merged, cv2.COLOR_BGR2RGB))


    hsv_container = st.beta_container()

    with hsv_container:
        description_col, space, img_col = st.beta_columns((2, .5, 4))

    with description_col:
        '''
        '''
    with img_col:
        # HSV experiments
        hsvimage = cv2.cvtColor(brg_merged, cv2.COLOR_RGB2HSV)
        hue, satr, vlue = cv2.split(hsvimage)
        hsvoutput = np.concatenate((hue, satr, vlue), axis=1)

        hsv_merged = cv2.merge([hue, satr, vlue])

        st.image(hsvoutput, caption="hsv comparison")
        st.image(hsv_merged, caption="hsv merged")
        save_hsv_img = st.button("save hsv image")
        if save_hsv_img:
            cv2.imwrite("./optos_tiff/{}_hsv-merged.png".format(file_object.name),
                        cv2.cvtColor(brg_merged, cv2.COLOR_BGR2RGB))


    eq_container = st.beta_container()

    with eq_container:
        description_col, space, img_col = st.beta_columns((2, .5, 4))

    with description_col:
        '''
        '''
    with img_col:
        # Equalized Images
        red_eq = cv2.equalizeHist(red,dst=None)
        blue_eq = cv2.equalizeHist(blue,dst=None)
        green_eq = cv2.equalizeHist(green, dst=None)
        eq_compare = np.concatenate((red_eq,green_eq,blue_eq),axis=1)

        eq_merged = cv2.merge((red_eq, green_eq, blue_eq))

        st.image(eq_compare, caption="histogram eq comparison")
        st.image(eq_merged, caption="histo eq output")
        save_eq_img = st.button("save eq image")
        if save_eq_img:
            cv2.imwrite("./optos_tiff/{}_eq-merged.png".format(file_object.name),
                        cv2.cvtColor(brg_merged, cv2.COLOR_BGR2RGB))



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
