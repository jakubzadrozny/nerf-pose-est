import numpy as np


class Transform:
    def __init__(self, R, t):
        self.R = R
        self.t = t

    @property
    def inverse(self):
        RT = self.R.T
        return Transform(RT, -RT @ self.t)

    @property
    def homogeneous(self):
        M = np.zeros((4, 4))
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        M[3, 3] = 1
        return M

    def __mul__(self, other):
        return Transform(self.R @ other.R,
                         self.t + self.R @ other.t)

    def __str__(self):
        return "R = " + str(self.R) + "\nt=" + str(self.t)

    __repr__ = __str__
