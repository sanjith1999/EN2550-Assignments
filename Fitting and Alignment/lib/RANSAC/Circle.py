import numpy as np
import math


class Circle(object):
    """Represents a Circle model using its center point and radius"""

    def __init__(self, center_x: float, center_y: float, radius: float):
        self.X: float = center_x
        self.Y: float = center_y
        self.R: float = radius
        pass

    def __str__(self):
        display = ("(%f, %f) R=%f" % (self.X, self.Y, self.R))
        return display

    @classmethod
    def GenerateModelFrom3Points(cls, points):
        [p1, p2, p3] = points
        x1, x2, x3 = p1[0], p2[0], p3[0]
        y1, y2, y3 = p1[1], p2[1], p3[1]
        c = (x1 - x2) ** 2 + (y1 - y2) ** 2
        a = (x2 - x3) ** 2 + (y2 - y3) ** 2
        b = (x3 - x1) ** 2 + (y3 - y1) ** 2
        s = 2 * (a * b + b * c + c * a) - (a * a + b * b + c * c)

        px = (a * (b + c - a) * x1 + b * (c + a - b) * x2 + c * (a + b - c) * x3) / s
        py = (a * (b + c - a) * y1 + b * (c + a - b) * y2 + c * (a + b - c) * y3) / s
        ar = a ** 0.5
        br = b ** 0.5
        cr = c ** 0.5
        r = ar * br * cr / ((ar + br + cr) * (-ar + br + cr) * (ar - br + cr) * (ar + br - cr)) ** 0.5

        circ = Circle(px, py, r)

        return circ

    @staticmethod
    def randy_bullock_fit(points):
        N = len(points)
        X, Y = points[:, 0], points[:, 1]
        x_bar, y_bar = np.mean(X), np.mean(Y)
        U, V = X - x_bar, Y - y_bar
        s_uu, s_uv, s_vv = np.sum(U * U), np.sum(U * V), np.sum(V * V)
        s_uuu, s_uvv, s_vuu, s_vvv = np.sum(U * U * U), np.sum(U * V * V), np.sum(V * U * U), np.sum(V * V * V)
        A = np.array([[s_uu, s_uv], [s_uv, s_vv]])
        B = np.array([[1 / 2 * (s_uuu + s_uvv)], [1 / 2 * (s_vvv + s_vuu)]])
        [[u_c], [v_c]] = np.linalg.inv(A) @ B
        alpha = math.pow(u_c, 2) + math.pow(v_c, 2) + (s_uu + s_vv) / N
        r = math.sqrt(alpha)
        c_x, c_y = u_c + x_bar, v_c + y_bar
        circ = Circle(c_x, c_y, r)
        return circ

    @staticmethod
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

    @staticmethod
    def inlier_counter(points, circle, threshold):
        """
        Calculating number of inliers lie at a given threshold
        :param points: data set
        :param circle: Circle Model
        :param threshold:
        :return: Count of Inliers
        """
        center_point = [circle.X, circle.Y]
        radius = circle.R
        distances = Circle.euclidean_distance(center_point, [points[:, 0], points[:, 1]])
        count = np.count_nonzero(np.logical_and(radius - threshold < distances, distances < radius + threshold))
        return count

    @staticmethod
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
        distances = Circle.euclidean_distance(center_point, [points[:, 0], points[:, 1]])
        inliers = points[np.logical_and(radius - threshold < distances, distances < radius + threshold)]
        outliers = points[np.logical_not(np.logical_and(radius - threshold < distances, distances < radius + threshold))]
        return inliers, outliers
