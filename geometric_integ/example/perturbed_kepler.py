from geometric_integ.integrate_method.projection_method import imp_mid_proj,exp_euler_proj
from geometric_integ.integrate_method.simple_numeric import implicit_midpoint
from geometric_integ.model_data.common_models import PerturbedKepler
from geometric_integ.plot_tools import draw_numeric2d, hamiltonian_energy_loss2d

import numpy as np
import matplotlib.pyplot as plt

def pkepler_test():
    h = 0.03; epoch = 10000; t0 = 0
    y0 = np.array([0, 2, 0.4, 0], dtype = np.float64)
    model_drawm = PerturbedKepler(t0, y0, h, epoch)
    H0 = model_drawm.H0

    class ManifoldH:
        def g(self, y):
            return model_drawm.H(y) - H0

        def Jg(self, y):
            return  model_drawm.JH(y)

    manifoldh = ManifoldH()

    plt.figure(figsize=(20, 10))
    plt.subplot(221)
    res_imp_proj = imp_mid_proj(model_drawm, manifoldh)
    draw_numeric2d(res_imp_proj, color='r', label = 'implict mid point proj')
    plt.legend()

    plt.subplot(222)
    res_imp_mid = implicit_midpoint(model_drawm)
    draw_numeric2d(res_imp_mid, color = 'r', label = 'implict mid point without proj')
    plt.legend()

###############################################能量绘制##############################################
    model_drawe1 = PerturbedKepler(t0, y0, h, epoch, data_range='all')
    model_drawe2 = PerturbedKepler(t0, y0, h, epoch, data_range='all')

    res_imp_proje = imp_mid_proj(model_drawe1, manifoldh)
    res_imp_mide = implicit_midpoint(model_drawe2)

    plt.subplot(223)
    hamiltonian_energy_loss2d(model_drawe1, res_imp_proje, color='b', label='implict mid proj energy')
    plt.legend()

    plt.subplot(224)
    hamiltonian_energy_loss2d(model_drawe2, res_imp_mide, color='g', label='implict mid point energy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    pkepler_test()
