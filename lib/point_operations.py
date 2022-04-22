import cv2 as cv
import numpy as np


def image_brighten(image, shift):
    """
    Increasing Brightness using Loop
    --------------------------------
    - This is Computationally expensive operation
    + Instead it is normally done in numpy which make use of vectorization and broadcasting
    """
    h = image.shape[0]
    w = image.shpae[1]
    result = np.zeros(image.shape, image.dtype)
    for i in range(h):
        for j in range(w):
            no_overflow = True if image[i, j]+shift < 255 else False
            result[i, j] = image[i, j] if no_overflow else 255
    return result


def equlize_histogram_(image):
    """ Equlizing Histogram of an image same as functionality of equalizeHist function """
    max_value=np.iinfo(image.dtype).max
    hist_image=cv.calcHist([image],[0],None,[max_value+1],[0,max_value+1])
    pdf=hist_image/np.sum(hist_image)
    cdf=np.cumsum(pdf)
    t_equalization=max_value*cdf
    return cv.LUT(image,t_equalization)



    