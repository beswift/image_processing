import cv2
import numpy as np


# load image as grayscale
def mask_fundus(image, filename):

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
    result_filename = "masked_"+filename
    # save resulting masked image
    cv2.imwrite(result_filename, result)
    cv2.imshow(result_filename,result)
    cv2.waitKey(0)
    return result

def countour_mask(image, filename):

   # find countours
    _, contours, _ = cv2.findContours(...)  
    idx = ...  # The index of the contour that surrounds your object
    mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, idx, 255, -1)  # Draw filled contour in mask
    out = np.zeros_like(img)  # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]

    # Show the output image
    cv2.imshow('Output', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()