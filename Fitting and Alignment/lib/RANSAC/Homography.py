import numpy as np
from .Circle import Circle


class Homography:
    """ Represents a Homography using the Homography matrix"""

    def __init__(self, H):
        self.H = H

    def __str__(self):
        display = ("%f\t%f\t%f \n%f\t%f\t%f \n%f\t%f\t%f" % (
            self.H[0][0], self.H[0][1], self.H[0][2], self.H[1][0], self.H[1][1], self.H[1][2], self.H[2][0], self.H[2][1], self.H[2][2]))
        return display

    @classmethod
    def generate_homography(cls, m_points):
        data_size = m_points.shape[0]
        p_i = np.hstack([m_points[:, 0], np.ones((m_points.shape[0], 1))])
        pd_i = m_points[:, 1]

        A = np.hstack([np.zeros((data_size, 3)), -p_i, pd_i[:, 0].reshape((data_size, 1)) * p_i, p_i, np.zeros((data_size, 3)), pd_i[:, 0].reshape((data_size, 1)) * p_i]).reshape(
            (2 * data_size, 9))
        U = A.T @ A
        W, V = np.linalg.eig(U)
        ev_corresponding_to_smallest_ev = V[:, np.argmin(W)]

        h_trans = Homography(np.array(ev_corresponding_to_smallest_ev.reshape((3, 3))))
        return h_trans

    @staticmethod
    def inlier_counter(m_coordinates, transformation, threshold):
        """
        Calculating number of inliers lie at a given threshold
        :param m_coordinates: list of matching points [[[x1,y1],[x2,y2]]]
        :param transformation: Homographic Transformation
        :param threshold: Expected Threshold
        :return: Number of Inliers
        """
        data_size = m_coordinates.shape[0]
        p_i = np.hstack([m_coordinates[:, 0], np.ones((data_size, 1))])
        p_i_3D = transformation.H @ (p_i.reshape((data_size, 3, 1)))
        p_i_3D = p_i_3D.reshape((data_size, 3))
        normalizer = p_i_3D[:, 2].reshape((data_size, 1))
        p_i_2D = p_i_3D[:, :2] / normalizer
        pd_i_2D = m_coordinates[:, 1]
        distances = Circle.euclidean_distance([p_i_2D[:, 0], p_i_2D[:, 1]], [pd_i_2D[:, 0], pd_i_2D[:, 1]])
        count = np.count_nonzero(distances < threshold)
        return count

    @staticmethod
    def inliers_outliers(m_coordinates, transformation, threshold):
        """
        Calculating Inliers
        :param m_coordinates:
        :param transformation:
        :param threshold:
        :return:
        """
        data_size = m_coordinates.shape[0]
        p_i = np.hstack([m_coordinates[:, 0], np.ones((data_size, 1))])
        p_i_3D = transformation.H @ (p_i.reshape((data_size, 3, 1)))
        p_i_3D = p_i_3D.reshape((data_size, 3))
        normalizer = p_i_3D[:, 2].reshape((data_size, 1))
        p_i_2D = p_i_3D[:, :2] / normalizer
        pd_i_2D = m_coordinates[:, 1]
        distances = Circle.euclidean_distance([p_i_2D[:, 0], p_i_2D[:, 1]], [pd_i_2D[:, 0], pd_i_2D[:, 1]])
        inliers = m_coordinates[distances < threshold]
        return inliers
