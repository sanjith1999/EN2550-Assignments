class Point:
    """Represents a point in 2 dimensional space"""

    def __init__(self, x, y):
        self.X = x
        self.Y = y

    # Calculating Euclidian Distance point1-> point2
    @staticmethod
    def euclidean_distance(point1, point2):
        squared = (point1.X - point2.X) ** 2 + (point1.Y - point2.Y) ** 2
        r = squared ** 0.5
        return r
