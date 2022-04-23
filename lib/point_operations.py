import cv2 as cv
import numpy as np
import math


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


def zoom(image, scaling_factor, method):
    """ 
    Zoom Image
    ----------
    image : ndarray 
    scaling factor : int or float 
    method : 'NN'-> Nearest Neighbour & 'BI' -> Bilinear Interpolation
    """
    img = image
    sf = scaling_factor

    # DETERMINING DIAMENSIONS AND GENERATING AN EMPTY VERTOR TO STORE ZOOMED IMAGE
    if len(img.shape) == 2: 
        zoomedImgDims = [int(dim*sf) for dim in img.shape]
    else:  
        zoomedImgDims = [int(dim*sf) for dim in img.shape]
        zoomedImgDims[2] = 3
    # declaring an empty array to store values
    zoomedImg = np.zeros(zoomedImgDims, dtype=img.dtype)

# -------------------------  NEAREST NEIGHBOUR   ---------------------------------#
    if method == 'NN':
        for row in range(zoomedImg.shape[0]):
            source_row = min(round(row/sf), img.shape[0]-1)
            for column in range(zoomedImg.shape[1]):
                source_column = min(round(column/sf), img.shape[1]-1)

                #FOR GRAY IMAGE
                if len(img.shape) == 2:
                    zoomedImg[row][column] = img[source_row][source_column]
                
                #FOR COLOR IMAGE
                else:
                    for channel in range(3):
                        zoomedImg[row][column][channel] = \
                            img[source_row][source_column][channel]
# -------------------------BILINEAR INTERPOLATION---------------------------------#
    if method == 'BI':
        for row in range(zoomedImg.shape[0]):
            row_position = row/sf
            row_below = math.floor(row_position)
            row_up = min(math.ceil(row_position),img.shape[0]-1)
            for column in range(zoomedImg.shape[1]):
                column_position = column/sf
                column_previous = math.floor(column_position)
                column_next = min(math.ceil(column_position),img.shape[1]-1)
                delta_row = row_position - row_below
                delta_column = column_position - column_previous

                #FOR GRAY IMAGE
                if len(img.shape) == 2:  
                    interVal1 = img[row_below][column_previous]*(1-delta_row)\
                        + img[row_up][column_previous]*(delta_row)
                    interVal2 = img[row_below][column_next]*(1-delta_row)\
                        + img[row_up][column_next]*(delta_row)
                    zoomedImg[row][column] = (interVal1*(1-delta_column)
                                              + interVal2*(delta_column)).astype('uint8')
                #FOR COLOR IMAGE
                else:  
                    for channel in range(3):
                        interVal1 = img[row_below][column_previous][channel]*(1-delta_row)\
                            + img[row_up][column_previous][channel]*(delta_row)
                        interVal2 = img[row_below][column_next][channel]*(1-delta_row)\
                            + img[row_up][column_next][channel]*(delta_row)
                        zoomedImg[row][column][channel] = (interVal1*(1-delta_column)
                                                           + interVal2*(delta_column)).astype('uint8')
    return zoomedImg

    