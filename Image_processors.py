import cv2
import numpy as np


# load image as grayscale
def mask_fundus(image, filename):
    cv2.imshow("test" + filename, image)
    cv2.waitKey(0)
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
    cv2.imshow(result_filename, result)
    cv2.waitKey(0)
    return result


def display_contours(image, contours, color = (255, 0, 0), thickness = -1, title = None):
    # Contours are drawn on the original image, so let's make a copy first
    imShow = image.copy()
    for i in range(1, len(contours)):
        cv2.drawContours(imShow, contours, i, color, thickness)
    cv2.imshow(title,imShow)
    cv2.waitKey()
    cv2.destroyAllWindows()

def countour_mask(image, filename):
    #todo find a way to autoset this threshold
    thresh = 45
    color = (255,0,0)
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
    for i in range(20,len(contours)):
        # Let's see what we've got:
        cv2.drawContours(image, contours, i, color, thickness)
    cv2.imshow('contour_image'+filename, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("{:d} points".format(len(np.vstack(np.array(contours)))))
    # Now let's get only what we need out of it
    hull_contours = cv2.convexHull(np.vstack(np.array(contours)))
    hull = np.vstack(hull_contours)
    print("{:d}stack arrays".format(len(np.vstack(np.array(hull)))))

    # we only get one contour out of it, let's see it
    title = "display_contour"+filename
    display_contours(image, [hull], thickness=3, color=(0, 255, 0),title=title )




