import math
import numpy as np
import cv2 as cv


def d_norm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma):
    if sigma < 0:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel_1D = np.zeros(size)
    for i in range(size):
        kernel_1D[i] = d_norm(i - size // 2, 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1 / kernel_2D.max()

    return kernel_2D


def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, -1)
    return filter_2d(image, kernel)


def detect(r, c, output, weak):
    if output[r, c] == weak:
        if 255 in [output[r - 1, c - 1], output[r - 1, c], output[r - 1, c + 1], output[r, c + 1], output[r + 1, c + 1], output[r + 1, c],
                   output[r + 1, c - 1], output[r, c - 1]]:
            output[r, c] = 255
        else:
            output[r, c] = 0
    return output


def filter_2d(image, kernel, is_convolve=True):
    """ 
    Convolution and Correlation operations
    image, kernel : ndarray
    is_convolve : Whether the operation is convolution(default) or correlation
    """
    # Color Image -> Gray Image
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Validating Kernel
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    kernel_h, kernel_w = math.floor(
        kernel.shape[0] / 2), math.floor(kernel.shape[1] / 2)
    image_h, image_w = image.shape
    image_float = image.astype('float')
    filtered = np.zeros(image.shape, 'float')

    if is_convolve:
        kernel = np.rot90(kernel, 2)

    for i in range(kernel_h, image_h - kernel_h):
        for j in range(kernel_w, image_w - kernel_w):
            filtered[i, j] = np.dot(image_float[i - kernel_h: i + kernel_h + 1,
                                    j - kernel_w: j + kernel_w + 1].flatten(), kernel.flatten())

    return filtered


def sobel_edge_detection(image, convert_to_degree=False):
    """
    Sobel Edge Detection
    --------------------
    image : ndarray
    convert_to_degrees : False by default
    returns -> gradient image, gradient direction
    """
    sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # np.flip((sobel_v.T),axis=0)

    image_x = filter_2d(image, sobel_v)
    image_y = filter_2d(image, sobel_h)

    gradient_magnitude = np.sqrt(np.square(image_x) + np.square(image_y))
    gradient_magnitude *= 255 / gradient_magnitude.max()

    gradient_direction = np.arctan2(image_y, image_x)

    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180

    return gradient_magnitude, gradient_direction


def non_max_suppression(gradient_magnitude, gradient_direction):
    """
    Non Maximum Suppression
    ---------------------
    gradient_magnitude : ndarray
    gradient_direction : ndarray
    return : Suppressed Gradient Magnitude
    """

    # Validating Input
    assert gradient_magnitude.shape == gradient_direction.shape

    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180
    count = 0

    # Approach
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
    return output


def threshold(image, low, high, weak=50):
    """
    Threshold
    ---------
    image : Gradient Magnitude
    low : Lower Threshold for weak edges
    high : higher Threshold for strong edges
    weak : 50 by default
    """

    output = np.zeros(image.shape)
    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    return output


def hysteresis(image, weak=50):
    """
    Hysteresis
    ---------
    image : ndarray (Threshold one)
    weak : value used for denoting weak edges(50 by default)
    """
    image_row, image_col = image.shape

    # Checking from the top
    top_to_bottom = image.copy()
    for row in range(1, image_row):
        for col in range(1, image_col):
            top_to_bottom = detect(row, col, top_to_bottom, weak)

    # Checking from the bottom
    bottom_to_top = image.copy()
    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            bottom_to_top = detect(row, col, bottom_to_top, weak)

    # Checking from the left
    right_to_left = image.copy()
    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            right_to_left = detect(row, col, right_to_left, weak)

    # Checking from the right
    left_to_right = image.copy()
    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            left_to_right = detect(row, col, left_to_right, weak)
    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    final_image[final_image > 255] = 255
    return final_image


def canny_detector(image, low, high):
    blurred_image = gaussian_blur(image, kernel_size=9)
    grad_mag, grad_dir = sobel_edge_detection(blurred_image, convert_to_degree=True)
    narrowed = non_max_suppression(grad_mag, grad_dir)
    t_image = threshold(narrowed, low, high, weak=50)
    return hysteresis(t_image, weak=50)
