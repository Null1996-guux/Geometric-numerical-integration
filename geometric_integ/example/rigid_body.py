from geometric_integ.integrate_method.simple_numeric import explicit_euler, implicit_midpoint
from geometric_integ.model_data.common_models import RigidBody
import numpy as np
from mayavi import mlab

# test of explicit euler method and implicit midpoint method on rigid body problem
def rigid_body_test1():
    t0 = 0
    y0 = np.array([np.cos(-2.1), 0, np.sin(2.1)])
    h = [0.1, 0.3]
    epoch = [320, 320]

    model_exper = RigidBody(t0, y0, h[0], epoch[0])
    model_impmpt = RigidBody(t0, y0, h[1], epoch[1])

    H0, F0 = model_exper.H0, model_exper.F0
    print('H0 = %f  F0 = %f' % (H0, F0))

    r = np.sqrt(F0)

    exper_res = explicit_euler(model_exper)
    impmpt_res = implicit_midpoint(model_impmpt)
    res = [exper_res, impmpt_res]

    # visuable soluiton
    colors = [(1, 0, 0), (0, 0, 1)]
    scales = [0.02, 0.03]
    mlab.figure(bgcolor=(1, 1, 1))
    for i in range(len(res)):
        x_plt, y_plt, z_plt = [], [], []
        for nume_solve in res[i]:
            x_plt.append(nume_solve[0])
            y_plt.append(nume_solve[1])
            z_plt.append(nume_solve[2])
        mlab.points3d(x_plt, y_plt, z_plt, color = colors[i], scale_factor = scales[i] )

    dtheta, dphi = np.pi / 250, np.pi / 250
    [theta, phi] =  np.mgrid[
        0 : 2 * np.pi + 1.5 * dtheta: dtheta,
        0 : np.pi + 1.5 * dphi : dphi
    ]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    mlab.mesh(x, y ,z, color = (1, 1, 0))

    mlab.show()


if __name__ == '__main__':
    rigid_body_test1()