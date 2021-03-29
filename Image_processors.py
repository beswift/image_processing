import cv2
import numpy as np
from io import BytesIO
from tifffile import imread,imwrite,imshow
import streamlit as st
from PIL import Image
import os


def get_fundus_mask(img):

    b,g,r = cv2.split(img)
    #Threshold to red values
    ret, img_threshold = cv2.threshold(r, 10, 255, cv2.THRESH_BINARY)
    # Create a circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # erode the kernel to isolate the most red area
    erode_img = cv2.erode(img_threshold, kernel, iterations=1)
    # Display results
    st.image(erode_img, caption="image erode")
    # Return result
    return erode_img


def circle_mask(image):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    centerx = int((image.shape[1])/2)
    centery = int((image.shape[0])/2)
    radius = int((image.shape[0])/1.7)
    cv2.circle(mask, (centerx,centery), radius, 255,-1)
    masked = cv2.bitwise_and(image,image, mask=mask)
    # show the output images
    #st.image(mask, "circular mask")
    #st.image(masked, "masked image")
    return masked

def mask_fundus(image, filename):
    # cv2.imshow("test" + filename, image)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold input image using otsu thresholding as mask and refine with morphology
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # put thresh into
    result = image.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    result_filename = "masked_" + filename
    # save resulting masked image
    cv2.imwrite(result_filename, result)
    # cv2.imshow(result_filename, result)
    # cv2.waitKey(0)
    return result


def display_contours(image, contours, color=(255, 0, 0), thickness=-1, title=None):
    # Contours are drawn on the original image, so let's make a copy first
    imShow = image.copy()
    for i in range(1, len(contours)):
        cv2.drawContours(imShow, contours, i, color, thickness)
    cv2.imshow(title, imShow)
    cv2.waitKey()
    cv2.destroyAllWindows()


def countour_mask(image, filename):
    # todo find a way to autoset this threshold
    thresh = 45
    color = (255, 0, 0)
    thickness = 2
    # Searcing for the eye
    # Let's see how this works setp-by-step
    # convert to a one channel image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge finder
    edges = np.array([])
    edges = cv2.Canny(gray, thresh, thresh * 3, edges)
    cv2.imshow('edges' + filename, edges)
    ret, threshImg = cv2.threshold(gray, 0, 255, 0)
    cv2.imshow('thresh' + filename, threshImg)

    cv2.waitKey()
    cv2.destroyAllWindows()
    # Find contours
    # second output is hierarchy - we are not interested in it.
    contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(20, len(contours)):
        # Let's see what we've got:
        cv2.drawContours(image, contours, i, color, thickness)
    cv2.imshow('contour_image' + filename, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("{:d} points".format(len(np.vstack(np.array(contours)))))
    # Now let's get only what we need out of it
    hull_contours = cv2.convexHull(np.vstack(np.array(contours)))
    hull = np.vstack(hull_contours)
    print("{:d}stack arrays".format(len(np.vstack(np.array(hull)))))

    # we only get one contour out of it, let's see it
    title = "display_contour" + filename
    display_contours(image, [hull], thickness=3, color=(0, 255, 0), title=title)


def convert2jpg(image):
    with BytesIO() as f:
        image.save(f, format='JPEG')
        return f.getvalue()


####  Image Manipulation helpers

def create_tiff_layers(filename):
    blue_init = imread(filename, key=0)
    green = imread(filename, key=1)
    red = imread(filename, key=2)
    blue = np.resize(blue_init, red.shape)
    blue[blue == 0] = 0
    return blue, green, red

def generate_tiff_files(filename):
    st.write(filename)
    image_data = filename.read()
    image = imread(image_data)
    return image


# Pipelines

def get_images(imageList,images_folder,image_source):
    outputList = []
    for image in imageList:
        if image_source == "Pick Folder":
            filename = image
            image_path = os.path.join(images_folder, image)
            imageIn = cv2.imread(image_path)
            #imageIn = imageIn.astype(np.uint16)
            #imageIn - cv2.cvtColor(imageIn,cv2.COLOR_BGR2RGB)
            imageIn = np.array(imageIn)
            outputList.append((imageIn,filename))
        if image_source =="File Uploader":
            filename = image.name
            image = Image.open(image)
            image_arr = np.array(image)
            #imageIn = cv2.imread(image_arr)
            imageIn = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
            #imageIn = imageIn.astype(np.uint16)
            #imageIn = cv2.cvtColor(cvimage,cv2.COLOR_BGR2RGB)
            imageIn = np.array(imageIn)
            outputList.append((imageIn,filename))
    return outputList

def pre_stitch(image):
    masked = circle_mask(image)
    b,g,r = cv2.split(masked)
    clahe = cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8))
    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)
    able = cv2.merge((clahe_r, clahe_g, clahe_b))
    #st.image(able)
    return able

