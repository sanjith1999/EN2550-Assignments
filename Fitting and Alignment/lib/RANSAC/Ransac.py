import numpy as np
import math
from .Circle import Circle
from .Homography import Homography


class Ransac:
    """Represents RANSAC method """

    def __init__(self, samples, threshold: float, model: str):
        self.samples = samples
        self.threshold: float = threshold
        self.model: str = model

        self.best_model, self.model_points, self.iteration_done, self.inlier_count = self.fit_model(self.samples, self.model, self.threshold)
        pass

    def __str__(self):
        display = ("Number of Samples = %d \n Iterations Done = %d\n Inliers Count : %d" % (len(self.samples), self.iteration_done, self.inlier_count))
        return display

    def get_model(self):
        """
        Getting the Model
        :return: best_model,final_model,model points
        """
        inliers, _ = self.get_inliers_outliers()
        if self.model == 'circle':
            final_model = Circle.randy_bullock_fit(inliers)
        else:
            final_model = Homography.calcHomography(inliers)
        print("Final Model : \n", final_model)
        return self.best_model, final_model, self.model_points

    def get_inliers_outliers(self):
        """
        Visualizing Inliers and Outliers
        :return: Points corresponding to inliers and outliers
        """
        if self.model == 'circle':
            inliers, outliers = Circle.inliers_outliers(self.samples, self.best_model, self.threshold)
        else:
            inliers, outliers = Homography.inliers_outliers(self.samples, self.best_model, self.threshold), []
        return inliers, outliers

    @staticmethod
    def fit_model(data, model='circle', threshold=0):
        """
        Fitting Circle Using RANSAC Algorithm
        :param data: data_points
        :param model: Which model to fit('circle' by default or homo)
        :param threshold: Typical value is 1.96\sigma
        :return: best model ,model points, Number of iterations done, Number of samples, Inlier count for the best model
        """
        num_iterations = math.inf
        iterations_done = 0

        max_inlier_count = 0
        best_model = None
        model_points = None

        desired_prob = 0.95
        data_size = len(data)

        # Assigning model parameters
        if model == 'circle':
            num_sample = 3
        else:
            num_sample = 5

        while num_iterations > iterations_done:
            np.random.shuffle(data)
            sample_points = data[:num_sample]
            if model == 'circle':
                estimated_model = Circle.GenerateModelFrom3Points(sample_points)
                inlier_count = Circle.inlier_counter(data, estimated_model, threshold)
            else:
                estimated_model = Homography.calcHomography(sample_points)
                inlier_count = Homography.inlier_counter(data, estimated_model, threshold)

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model
                model_points = sample_points

                prob_outlier = 1 - inlier_count / data_size
                try:
                    num_iterations = math.log(1 - desired_prob) / math.log(1 - (1 - prob_outlier) ** num_sample)
                except:
                    pass
            iterations_done = iterations_done + 1

        return best_model, model_points, iterations_done, max_inlier_count
