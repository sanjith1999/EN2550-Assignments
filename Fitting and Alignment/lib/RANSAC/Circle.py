class Circle(object):
    """Represents a Circle model using its center point and radius"""

    def __init__(self, center_x: float, center_y: float, radius: float):
        self.X: float = center_x
        self.Y: float = center_y
        self.R: float = radius
        pass

    def __str__(self):
        display = ("X=%f Y=%f R=%f" % (self.X, self.Y, self.R))
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
