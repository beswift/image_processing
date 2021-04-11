import numpy as np
import cv2
from Image_processors import *
import streamlit as st
from collections import OrderedDict




# Harris corner detection:
def get_harris_corners(image, filename):
    img_gray = cv2.cv2tColor(image, cv2.COLOR_BGR2GRAY)
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
    gray = cv2.cv2tColor(masked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.9, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except:
        sift = cv2.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    print(" kps: {}, descriptors: {}".format(len(kps), descs.shape))
    rgb = cv2.cv2tColor(masked, cv2.COLOR_BGR2RGB)
    sift_image = cv2.drawKeypoints(rgb, kps, None, (0, 0, 255), 4)
    return sift_image,kps, descs


# Surf detection
def surf_features(image,filename):
    masked = circle_mask(image)
    gray = cv2.cv2tColor(masked, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    (kps, descs) = surf.detectAndCompute(gray, None)
    surf_image = cv2.drawKeypoints(masked, kps, None, (255, 0, 0), 4)
    print(" kps: {}, descriptors: {}".format(len(kps), descs.shape))
    surf_image_name = filename + " surf image"
    cv2.imshow(surf_image_name, surf_image)
    cv2.waitKey(0)


def compare_images(img1,img2):
    # read images
    img1 = cv2.cv2tColor(img1,cv2.COLOR_BGR2RGB)
    img2 = cv2.cv2tColor(img2,cv2.COLOR_BGR2RGB)

    grey1 = cv2.cv2tColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cv2tColor(img2, cv2.COLOR_BGR2GRAY)

    masked1 = circle_mask(grey1)
    masked2 = circle_mask(grey2)

    # sift
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(masked1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(masked2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[0:450], img2, flags=2)
    st.image(img3,caption='compared images')
    #cv2.imshow('match', img3)
    #cv2.waitKey(0)

def compare_images2(imageList):
    for img in imageList:
        # todo need to output a dict with a key:value like img:keypoints  or list with tuples like (img,keypoints)
        pass


EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv2.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv2.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv2.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv2.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv2.detail.ExposureCompensator_NO
def display_expos_comp(expos_comp_choices):
    st.write(expos_comp_choices)
    choice_name = expos_comp_choices
    return choice_name

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv2.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv2.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv2.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv2.detail_NoBundleAdjuster

FEATURES_FIND_CHOICES = OrderedDict()
try:
    cv2.xfeatures2d_SURF.create() # check if the function can be called
    FEATURES_FIND_CHOICES['surf'] = cv2.xfeatures2d_SURF.create
except (AttributeError, cv2.error) as e:
    print("SURF not available")
# if SURF not available, ORB is default
FEATURES_FIND_CHOICES['orb'] = cv2.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv2.xfeatures2d_SIFT.create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['brisk'] = cv2.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv2.AKAZE_create
except AttributeError:
    print("AKAZE not available")

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['gc_color'] = cv2.detail_GraphCutSeamFinder('COST_COLOR')
SEAM_FIND_CHOICES['gc_colorgrad'] = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
SEAM_FIND_CHOICES['dp_color'] = cv2.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv2.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv2.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv2.detail_AffineBasedEstimator

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

WAVE_CORRECT_CHOICES = ('horiz', 'no', 'vert',)

BLEND_CHOICES = ('multiband', 'feather', 'no',)



def stitch_images(imagesList,mode,confidenceThresh):

    if mode == "panorama":
        st.write("panorama stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    elif mode == "affine":
        st.write("affine stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)
    else:
        st.write("hmm, no mode?  stitching up like a robot")
        stitcher = cv2.Stitcher_create()
    stitcher.setPanoConfidenceThresh(confidenceThresh)
    stitched = stitcher.stitch(imagesList)
    return stitched


def get_matcher(gpu,matcher,match_conf,features,rangewidth):
    try_cuda = gpu
    matcher_type = matcher
    if match_conf is None:
        if features == 'orb':
            match_conf = 0.3
        else:
            match_conf = 0.65
    else:
        match_conf = match_conf
    range_width = rangewidth
    if matcher_type == "affine":
        matcher = cv2.detail_AffineBestOf2NearestMatcher(False, try_cuda,match_conf)
    elif range_width == -1:
        matcher = cv2.detail.BestOf2NearestMatcher_create(try_cuda,match_conf)
    else:
        matcher = cv2.detail.BestOf2NearestRangeMatcher_create(range_width,try_cuda, match_conf)
    return matcher

def adv_stitch_images(imagesList,mode,confidenceThresh,work_megapix,features, matcher, estimator, match_conf, ba, ba_refine_mask, save_graph, wave_correct, warp, blend, expos_comp, seam):

    if mode == "panorama":
        st.write("panorama stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    elif mode == "affine":
        st.write("affine stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)
    else:
        st.write("hmm, no mode?  stitching up like a robot")
        stitcher = cv2.Stitcher_create()
    stitcher.setPanoConfidenceThresh(confidenceThresh)
    stitched = stitcher.stitch(imagesList)
    return stitched


def adv_stitch_images(imagesList,mode,confidenceThresh,work_megapix,features, matcher, estimator, match_conf, ba, ba_refine_mask, save_graph, wave_correct, warp, blend, expos_comp, seam):

    if mode == "panorama":
        st.write("panorama stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    elif mode == "affine":
        st.write("affine stitching")
        stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)
    else:
        st.write("hmm, no mode?  stitching up like a robot")
        stitcher = cv2.Stitcher_create()
    stitcher.setPanoConfidenceThresh(confidenceThresh)
    stitched = stitcher.stitch(imagesList)
    return stitched

