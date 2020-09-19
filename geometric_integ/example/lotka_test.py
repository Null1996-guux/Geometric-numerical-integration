
from geometric_integ.integrate_method.simple_numeric import explicit_euler, implicit_euler, symplectic_euler
from geometric_integ.model_data.common_models import LotkaModel
from geometric_integ.plot_tools import my_draw_contour2d,draw_numeric2d
import numpy as np
import matplotlib.pyplot as plt

def test_lotka():
    t0 = [0, 0, 0]
    y0 = [np.array([2, 2]), np.array([4, 8]), np.array([6, 2])]
    h = [0.12, 0.12, 0.12]
    epoch = [100, 100, 100]

    methods = [explicit_euler, implicit_euler, symplectic_euler]
    labels = ['explicit euler res', 'implicit euler res', 'symplectic euler res']

    f_lotka = lambda u, v: np.log(u) - u + 2 * np.log(v) - v

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        model = LotkaModel(t0[i], y0[i], h[i], epoch[i])
        res = methods[i](model)
        my_draw_contour2d(f_lotka, (0.01, 8, 100), (0.01, 9, 100), curve_number = 20)
        draw_numeric2d(res, label= labels[i])
        plt.legend()
    plt.show()

if __name__ == '__main__':
    test_lotka()