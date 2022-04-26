import numpy as np
from .Circle import Circle
import math

def euclidean_distance(point1, point2):
    """
    Calculating the distance between two points
    :param point1: [x1,y1]
    :param point2: [x2,y2]
    :return: Distance between two points
    """
    squared = (point1[0] - point2[0]) ** 2 + (point1[0] - point2[1]) ** 2
    r = squared ** 0.5
    return r


def inlier_counter(points, circle, threshold):
    """
    Calculating number of inliers lie at a given threshold
    :param points: data set
    :param circle: Circle Model
    :param threshold:
    :return:
    """
    center_point = [circle.X, circle.Y]
    radius = circle.R
    distances = euclidean_distance(center_point, [points[:, 0], points[:, 1]])
    count = np.count_nonzero(np.logical_and(radius - threshold < distances, distances < radius + threshold))
    return count


def inliers_outliers(points, circle, threshold):
    """
    Calculating Inliers and Outliers
    :param points:
    :param circle:
    :param threshold:
    :return:
    """
    center_point = [circle.X, circle.Y]
    radius = circle.R
    distances = euclidean_distance(center_point, [points[:, 0], points[:, 1]])
    inliers = points[np.logical_and(radius - threshold < distances, distances < radius + threshold)]
    outliers = points[np.logical_not(np.logical_and(radius - threshold < distances, distances < radius + threshold))]
    return inliers, outliers


