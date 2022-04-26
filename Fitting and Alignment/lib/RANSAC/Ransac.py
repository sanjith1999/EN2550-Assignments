import numpy as np
import math
from .Circle import Circle
from .descriptors import inlier_counter, inliers_outliers


class Ransac:
    """Represents RANSAC method """

    def __init__(self, samples: [[float, float]], threshold: float):
        self.samples: [[float, float]] = samples
        self.threshold: float = threshold
        self.best_model, self.model_points, self.iteration_done, self.inlier_count = self.fit_circle(self.samples, self.threshold)
        pass

    def __str__(self):
        print("Model Parameters : ", self.best_model)
        display = ("Number of Samples = %d \n Iterations Done = %d\n Inliers Count : %d" % (len(self.samples), self.iteration_done, self.inlier_count))
        return display

    @staticmethod
    def fit_circle(data, threshold):
        """
        Fitting Circle Using RANSAC Algorithm
        :param data: data_points
        :param threshold: Typical value is 1.96\sigma
        :return: best circular model ,model points, Number of iterations done, Number of samples, Inlier count for the best model
        """
        num_iterations = math.inf
        iterations_done = 0
        num_sample = 3

        max_inlier_count = 0
        best_model = None
        model_points = None

        prob_outlier = 0.5
        desired_prob = 0.95

        data_size = len(data)

        while num_iterations > iterations_done:
            np.random.shuffle(data)
            sample_points = data[:num_sample]
            estimated_circle = Circle.GenerateModelFrom3Points(sample_points)
            inlier_count = inlier_counter(data, estimated_circle, threshold)

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_circle
                model_points = sample_points

                prob_outlier = 1 - inlier_count / data_size
                try:
                    num_iterations = math.log(1 - desired_prob) / math.log(1 - (1 - prob_outlier) ** num_sample)
                except:
                    pass
            iterations_done = iterations_done + 1

        return best_model, model_points, iterations_done, max_inlier_count

    def get_model(self):
        """
        Getting Circle Model
        :return: Circle(cls),model points
        """
        return self.best_model, self.model_points

    def get_inliers_outliers(self):
        """
        Visulizing Inliers and Outliers
        :return: Points corresponding to inliers and outliers
        """
        inliers, outliers = inliers_outliers(self.samples, self.best_model, self.threshold)
        return inliers, outliers
