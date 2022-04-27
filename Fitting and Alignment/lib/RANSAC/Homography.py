import numpy as np
from .Circle import Circle


class Homography:
    """ Represents a Homography using the Homography matrix"""

    def __init__(self, H):
        self.H = np.array(H)

    def __str__(self):
        display = ("%f\t%f\t%f \n%f\t%f\t%f \n%f\t%f\t%f" % (
            self.H[0][0], self.H[0][1], self.H[0][2], self.H[1][0], self.H[1][1], self.H[1][2], self.H[2][0], self.H[2][1], self.H[2][2]))
        return display

    @staticmethod
    def normalizePoints(points):
        """
        Normalizing the point cloud(pre-conditioning)
        """

        centroid = sum(points) / len(points)
        total_dist = sum(np.sqrt(np.sum(((points - centroid) ** 2), axis=1)))
        avg_dist = total_dist / len(points)
        scale = np.sqrt(2) / avg_dist
        xt, yt = centroid
        transform = np.array([[scale, 0, -xt * scale],
                              [0, scale, -yt * scale],
                              [0, 0, 1]])
        points = np.concatenate((points, np.ones((len(points), 1))), axis=1)

        normalized_points = transform.dot(points.T).T
        return transform, normalized_points

    @staticmethod
    def calcHomography(m_points):
        """
        The normalized DLT for 2D homographs.
        """
        p1, p2 = m_points[:, 0], m_points[:, 1]
        # Normalizing the points using predefined function
        T1, p1 = Homography.normalizePoints(p1)
        T2, p2 = Homography.normalizePoints(p2)
        # Initialising an array to keep the coefficient matrix
        A = np.zeros((2 * len(p1), 9))
        row = 0
        # Filling rows of the matrix according to the expressions
        for point1, point2 in zip(p1, p2):
            # Coefficients of the current row
            A[row, 3:6] = -point2[2] * point1
            A[row, 6:9] = point2[1] * point1
            # Coefficients of the next row
            A[row + 1, 0:3] = point2[2] * point1
            A[row + 1, 6:9] = -point2[0] * point1
            row += 2
        # Singular Value decomposition of A
        U, D, VT = np.linalg.svd(A)
        # unit singular vector corresponding to the smallest
        # singular value, is the solution h. That is last column of V.
        # i.e. Last row of the V^T
        h = VT[-1]
        # Reshaping to get 3x3 homography
        H = h.reshape((3, 3))
        # Renormalization
        H = np.linalg.inv(T2).dot(H).dot(T1)
        H = H / H[-1, -1]
        return Homography(H)

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
