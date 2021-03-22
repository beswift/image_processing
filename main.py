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
import tifffile

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
        func_image = file_uploader("compare")

    working_folder = st.selectbox("pick a working folder", os.listdir(os.getcwd()))
    images_folder = os.path.join(os.getcwd(), working_folder).lower()
    for image in glob.glob(('{}//*.*'.format(images_folder))):
        image_path = os.path.join(images_folder, image)
        st.write(image_path)
        imageIn = cv2.imread(image_path)
        sift_image, sift_image_name = sift_features(imageIn,image)
        st.image(sift_image,caption=sift_image_name)
        #gray = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)
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

    '''
    ## Process Tiff files:


    '''
    proc_container = st.beta_container()

    with proc_container:
        description_col, space, img_col = st.beta_columns((2, .5, 4))


    with description_col:
        tiff_folder = os.path.join(os.getcwd(),"optos_tiff")



        '''
        ###  First, pick a Tiff file to explore on the right
        
        '''
        #tiff_file = st.selectbox("pick a tiff file",options=os.listdir(tiff_folder))
        #st.write(tiff_file)
        # m_tiff = os.path.join(tiff_folder,tiff_file)

        '''
        ### Set options here
        '''
        brightness = st.slider("brightness:", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
        contrast = st.slider("contrast:",min_value=-10.0 , max_value=10.0 , value=1.0, step=0.1 )
        alpha = st.slider("select an Alpha Value:",min_value=0.0, max_value= 65535.0, value= 0.0, step=.5)
        beta = st.slider("select a Beta Value:", min_value=0.0, max_value=65535.0, value=65535.0,step =.5)
        #normtype = st.select_slider("select a norm type", options=[cv2.,8], value = 8)
        cliplimit = st.slider("select a clip limit", min_value=0.0, max_value = 100.0, value = 3.0, step=.1)
        tileGridSize = st.select_slider("select a tile grid size", options=[(1,1),(2,2),(3,3),(5,5),(8,8),(13,13),(21,21)])


        destination = None



        st.write("merge tiffs..")




    with img_col:

        # upload a file
        file_object = files_uploader("merge")
        tiff_name = file_object.name
        m_tiff = os.path.join(tiff_folder,tiff_name)

        # get file layers
        blue, green, red = create_tiff_layers(m_tiff)

        # Basic Comparison
        brg_compare = np.concatenate((blue, green, red), axis=1)

        # Merge Layers - rgb
        brg_merged = cv2.merge([red, green, blue])

        # Merged single channels
        red_merge = cv2.merge([red,red,red])
        blue_merge = cv2.merge([blue,blue,blue])
        green_merge = cv2.merge([green,green,green])

        merge_compare = np.concatenate((red_merge, green_merge, blue_merge), axis=1)

        merged_merged = cv2.merge([red_merge,green_merge,blue_merge])

        # CHLAE image channels
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tileGridSize)

        red_clahe = clahe.apply(red)
        blue_clahe = clahe.apply(blue)
        green_clahe = clahe.apply(green)

        clahe_compare = np.concatenate((red_clahe, green_clahe, blue_clahe), axis=1)

        clahe_merged = cv2.merge([red_clahe,green_clahe,blue_clahe])


        # HSV experiments
        #hsvimage = cv2.cvtColor(brg_merged, cv2.COLOR_BGR2HSV)
        #hue, satr, vlue = cv2.split(hsvimage)
        #hsvoutput = np.concatenate((hue, satr, vlue), axis=1)

        # Scale image channel

        blue_scaled = cv2.normalize(blue, dst=destination, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
        green_scaled = cv2.normalize(green, dst=destination, alpha=alpha, beta= beta, norm_type=cv2.NORM_MINMAX)
        red_scaled = cv2.normalize(red, dst=destination, alpha=alpha, beta = beta, norm_type=cv2.NORM_MINMAX)

        scaled_compare = np.concatenate((red_scaled, green_scaled, blue_scaled), axis=1)

        scaled_merge = cv2.merge([red_scaled, green_scaled, blue_scaled])

        # Equalized Images
        red_eq = cv2.equalizeHist(red,dst=None)
        blue_eq = cv2.equalizeHist(blue,dst=None)
        green_eq = cv2.equalizeHist(green, dst=None)
        eq_compare = np.concatenate((red_eq,green_eq,blue_eq),axis=1)

        eq_merged = cv2.merge((red_eq, green_eq, blue_eq))

        # Display Images

        bgr_display = st.image(brg_compare, caption="Layer Comparison")

        st.image(brg_merged, caption='output image')
        st.image(merge_compare, caption = 'merge compare')
        #st.image(merged_merged, caption="merged output")
        st.image(scaled_compare, caption="Scaled Comparison")
        st.image(clahe_compare, caption="clahe comparison")
        st.image(clahe_merged, caption="clahe output")
        st.image(eq_compare, caption="histogram eq comparison")
        st.image(eq_merged, caption="histo eq output")


        #st.image(hsvoutput, caption="Hsv Image Comparison")
        st.image(eq_merged, caption="equalized output")


        with TiffFile("optos_tiff/AC0090-20180912@093744-R2-P.tif") as tif:
            for page in tif.pages:
                image = page.asarray()
                #st.write(page.tags)
                st.image(image)
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
