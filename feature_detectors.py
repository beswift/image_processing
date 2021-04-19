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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def sift_features(image):
    masked = circle_mask(image)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.9, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except:
        sift = cv2.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    print(" kps: {}, descriptors: {}".format(len(kps), descs.shape))
    rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
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
    try:
        FEATURES_FIND_CHOICES['sift'] = cv2.SIFT_create
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

def get_compensator(expos_comp,expos_comp_nr_feeds,expos_comp_block_size):
    expos_comp_type = EXPOS_COMP_CHOICES[expos_comp]
    expos_comp_nr_feeds = expos_comp_nr_feeds
    expos_comp_block_size = expos_comp_block_size
    # expos_comp_nr_filtering = args.expos_comp_nr_filtering
    if expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS:
        compensator = cv2.detail_ChannelsCompensator(expos_comp_nr_feeds)
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv2.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator



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


def detail_stitch(images,work_megapix,seam_megapix,compose_megapix):
    input_images = images
    img_names = args.img_names
    print(img_names)
    work_megapix = args.work_megapix
    seam_megapix = args.seam_megapix
    compose_megapix = args.compose_megapix
    conf_thresh = args.conf_thresh
    ba_refine_mask = args.ba_refine_mask
    wave_correct = args.wave_correct
    if wave_correct == 'no':
        do_wave_correct = False
    else:
        do_wave_correct = True
    if args.save_graph is None:
        save_graph = False
    else:
        save_graph = True
    warp_type = args.warp
    blend_type = args.blend
    blend_strength = args.blend_strength
    result_name = args.output
    if args.timelapse is not None:
        timelapse = True
        if args.timelapse == "as_is":
            timelapse_type = cv2.detail.Timelapser_AS_IS
        elif args.timelapse == "crop":
            timelapse_type = cv2.detail.Timelapser_CROP
        else:
            print("Bad timelapse method")
            exit()
    else:
        timelapse = False
    finder = FEATURES_FIND_CHOICES[args.features]()
    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    for name in img_names:
        full_img = cv2.imread(cv2.samples.findFile(name))
        if full_img is None:
            print("Cannot read image ", name)
            exit()
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv2.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale,
                            interpolation=cv2.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv2.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv2.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        images.append(img)

    matcher = get_matcher(args)
    p = matcher.apply2(features)
    matcher.collectGarbage()

    if save_graph:
        with open(args.save_graph, 'w') as fh:
            fh.write(cv2.detail.matchesGraphAsString(img_names, p, conf_thresh))

    indices = cv2.detail.leaveBiggestComponent(features, p, 0.3)
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    for i in range(len(indices)):
        img_names_subset.append(img_names[indices[i, 0]])
        img_subset.append(images[indices[i, 0]])
        full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
    images = img_subset
    img_names = img_names_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(img_names)
    if num_images < 2:
        print("Need more images")
        exit()

    estimator = ESTIMATOR_CHOICES[args.estimator]()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = BA_COST_CHOICES[args.ba]()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        exit()
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    if do_wave_correct:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        rmats = cv2.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv2.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    seam_finder = SEAM_FIND_CHOICES[args.seam]
    seam_finder.find(images_warped_f, corners, masks_warped)
    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    timelapser = None
    # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
    for idx, name in enumerate(img_names):
        full_img = cv2.imread(name)
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])
        if abs(compose_scale - 1) > 1e-1:
            img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            img = full_img
        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv2.dilate(masks_warped[idx], None)
        seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0,
                              cv2.INTER_LINEAR_EXACT)
        mask_warped = cv2.bitwise_and(seam_mask, mask_warped)
        if blender is None and not timelapse:
            blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv2.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
            elif blend_type == "feather":
                blender = cv2.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        elif timelapser is None and timelapse:
            timelapser = cv2.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)
        if timelapse:
            ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
            timelapser.process(image_warped_s, ma_tones, corners[idx])
            pos_s = img_names[idx].rfind("/")
            if pos_s == -1:
                fixed_file_name = "fixed_" + img_names[idx]
            else:
                fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
            cv2.imwrite(fixed_file_name, timelapser.getDst())
        else:
            blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])
    if not timelapse:
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        cv2.imwrite(result_name, result)
        zoom_x = 600.0 / result.shape[1]
        dst = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dst = cv2.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        cv2.imshow(result_name, dst)
        cv2.waitKey()

    print("Done")