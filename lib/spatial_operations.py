import math
import numpy as np


def filter_2d(image, kernel,is_convolve=True):
    """ 
    Convolution and Correlation operations
    image, kernel : ndarray
    is_convolve : Whether the operation is convolution(default) or correlation
    """

    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    kernel_h, kernel_w = math.floor(kernel.shape[0]/2), math.floor(kernel.shape[1]/2)
    image_h, image_w = image.shape
    image_float = image.astype('float')
    filtered = np.zeros(image.shape, 'float')

    if is_convolve:
        kernel=np.rot90(kernel,2)

    for i in range(kernel_h, image_h - kernel_h):
        for j in range(kernel_w, image_w - kernel_w):
            filtered[i, j] = np.dot(image_float[i - kernel_h: i + kernel_h + 1,j - kernel_w: j + kernel_w + 1].flatten(), kernel.flatten())

    return filtered

def canny_edge_detector(image):