import numpy as np
import cv2
from Image_processors import *
import streamlit as st



# Harris corner detection:
def get_harris_corners(image, filename):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(img_gray)
    dst = cv2.cornerHarris(gray, 1, 3, 0.08)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    image[res[:, 1], res[:, 0]] = [0, 0, 255]
    image[res[:, 3], res[:, 2]] = [0, 255, 0]
    harris_image = filename + "harris"
    cv2.imshow(harris_image, image)
    cv2.waitKey(0)


# Sift detection
def sift_features(image):
    masked = circle_mask(image)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except:
        sift = cv2.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    print(" kps: {}, descriptors: {}".format(len(kps), descs.shape))
    sift_image = cv2.drawKeypoints(gray, kps, None, (255, 0, 0), 4)
    return sift_image


# Surf detection
def surf_features(image, filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    (kps, descs) = surf.detectAndCompute(gray, None)
    surf_image = cv2.drawKeypoints(gray, kps, None, (255, 0, 0), 4)
    print(" kps: {}, descriptors: {}".format(len(kps), descs.shape))
    surf_image_name = filename + " surf image"
    cv2.imshow(surf_image_name, surf_image)
    cv2.waitKey(0)


def compare_images():
    # read images
    img1 = cv2.cvtColor(cv2.imread('test_images//A-1.jpg'),cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('test_images//A-2.jpg'),cv2.COLOR_BGR2RGB)

    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    masked1 = circle_mask(grey1)
    masked2 = circle_mask(grey2)


    # sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(masked1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(masked2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[420:450], img2, flags=2)
    st.image(img3,caption='compared images')
    #cv2.imshow('match', img3)
    #cv2.waitKey(0)

def compare_images2(imageList):
    for img in imageList:
        # todo need to output a dict with a key:value like img:keypoints  or list with tuples like (img,keypoints)
        pass



def stitch_images(imagesList,mode):
    if mode == "panorama":
        st.write("panorama stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    elif mode == "affine":
        st.write("affine stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)
    else:
        st.write("hmm, no mode?  stitching up like a robot")
        stitcher = cv2.Stitcher_create()
    stitched = stitcher.stitch(imagesList)
    return stitched

