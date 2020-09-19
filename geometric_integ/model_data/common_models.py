import numpy as np
from geometric_integ.model_data.model import DataModel

# Lotka Model
class LotkaModel(DataModel):
    def __init__(self, t0, y0, h, epoch, data_range = 'all'):
        DataModel.__init__(self, t0, y0, h, epoch, data_range)

    def f(self, y):
        return np.array([y[0] * (y[1] - 2), y[1] * (1 - y[0])])

# Kepler Mpdel
class KeplerModel(DataModel):
    def __init__(self, t0, y0, h, epoch, data_range = [2, 4]):
        DataModel.__init__(self, t0, y0, h, epoch, data_range)

    def f(self, y):
        p1, p2, q1, q2  = y[0], y[1], y[2], y[3]
        return np.array([-q1 / (q1 ** 2 + q2 ** 2) ** 1.5, -q2 / (q1 ** 2 + q2 ** 2) ** 1.5, p1, p2])

    def f_stormer(self, y):
        return np.array([-y[0] / (y[0] ** 2 + y[1] ** 2) ** 1.5, -y[1] / (y[0] ** 2 + y[1] ** 2) ** 1.5])

    # Hamiltonian total energy
    def H(self, y):
        return 0.5 * (y[0] ** 2 + y[1] ** 2) - 1 / np.sqrt(y[2] ** 2 + y[3] ** 2)

class OuterSolar(DataModel):
    def __init__(self, t0, y0, h, epoch, data_range):
        DataModel.__init__(self, t0, y0, h, epoch, data_range)
        pass


class RigidBody(DataModel):
    def __init__(self, t0, y0, h, epoch, data_range = 'all', I = np.array([2, 1, 2/3])):
        DataModel.__init__(self, t0, y0, h, epoch, data_range)
        self.I = I
        self.H0 = self.H(self.y0)
        self.F0 = self.F(self.y0)

    def f(self, y):
        mat_tmp = np.array([
            [0, y[2] / self.I[2], -y[1] / self.I[1]],
            [-y[2] / self.I[2], 0, y[0] / self.I[0]],
            [y[1] / self.I[1], -y[0] / self.I[0], 0]
        ])
        return np.dot(mat_tmp, y.reshape(-1, 1)).flatten()

    def F(self, y):
        return np.linalg.norm(y) ** 2

    def H(self, y):
        return 0.5 * (y[0] ** 2 / self.I[0] + y[1] ** 2 / self.I[1] + y[2] ** 2 / self.I[2])


class PerturbedKepler(DataModel):
    def __init__(self, t0, y0, h, epoch, data_range = [2, 4]):
        DataModel.__init__(self, t0, y0, h, epoch, data_range)
        self.H0 = self.H(y0)
        self.L0 = self.L(y0)

    def H(self, x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2) - 1 / (np.sqrt(x[2] ** 2 + x[3] ** 2)) - 0.0025 / (
            np.sqrt((x[2] ** 2 + x[3] ** 2) ** 3))

    def L(self, x):
        return x[2] * x[1] - x[3] * x[0]

    def f(self, x):
        return np.array([-self.JH(x)[2], -self.JH(x)[3], self.JH(x)[0], self.JH(x)[1]])

    def JH(self, x):
        return np.array([
            x[0], x[1],
            (x[2] / (x[2] ** 2 + x[3] ** 2) ** 1.5) * (1 + 0.0075 / (x[2] ** 2 + x[3] ** 2)),
            (x[3] / (x[2] ** 2 + x[3] ** 2) ** 1.5) * (1 + 0.0075 / (x[2] ** 2 + x[3] ** 2))
        ])

    def JL(self, x):
        return np.array([-x[3], x[2], x[1], -x[0]])

    def both(self, x):
        return np.array([[self.H(x), self.L(x)]])

    def Jboth(self, x):
        return np.vstack((self.JH(x), self.JL(x)))


