import numpy as np


class Transform:
    def __init__(self, *args):
        if len(args) == 1:
            # homogeneous matrix case
            M = args[0]
            assert M.shape == (4, 4)
            MM = M / M[3, 3]
            self.R = MM[:3, :3]
            self.t = MM[:3, 3]
        elif len(args) == 2:
            # R, t case
            self.R = args[0]
            self.t = args[1]
        else:
            raise NotImplementedError

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


class ZRotation(Transform):
    def __init__(self, angle):
        angle = np.radians(angle)
        cos = np.cos(angle)
        sin = np.sin(angle)
        R = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1],
        ])
        super().__init__(R, np.zeros(3))
