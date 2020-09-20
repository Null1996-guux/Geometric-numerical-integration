import numpy as np
from scipy.optimize import fsolve

# simple newton method of scalar manifold
def simple_newton_sca(manifold, x_tilde, epoch = 3):
    grad_inv = np.dot(manifold.Jg(x_tilde).T, manifold.Jg(x_tilde))**(-1)
    lam = 0
    for i in range(epoch):
        lam -= grad_inv * manifold.g(x_tilde + lam * manifold.Jg(x_tilde))
    res = x_tilde + lam * manifold.Jg(x_tilde)
    return res

def exp_euler_proj(model, manifold, proj_dim = 1):
    y_cur = model.y0
    for i in range(model.epoch):
        y_tilde = y_cur + model.h * model.f(y_cur)
        if proj_dim == 1:
            y_cur = simple_newton_sca(manifold, y_tilde)
        model.add_res(y_cur)
    return model.get_res()

def imp_mid_proj(model, manifold, proj_dim = 1):
    y_cur = model.y0
    for i in range(model.epoch):
        fc = lambda x: y_cur + model.h * model.f((y_cur + x) / 2) - x
        y_tilde = fsolve(fc, y_cur)
        if proj_dim == 1:
            y_cur = simple_newton_sca(manifold, y_tilde)
        model.add_res(y_cur)
    return model.get_res()

def symp_euler_proj(model, manifold, proj_dim = 1):
    tt_l = len(model.y0)
    u_cur, v_cur = model.y0[0: tt_l // 2], model.y0[tt_l // 2:]
    a = lambda u, v: model.f(np.hstack((u, v)))[0: tt_l // 2]
    b = lambda u, v: model.f(np.hstack((u, v)))[tt_l // 2:]
    for i in range(model.epoch):
        fc = lambda x: np.array([
            x[0: tt_l // 2] - u_cur - model.h * a(x[0: tt_l // 2], v_cur),
            x[tt_l // 2:] - v_cur - model.h * b(x[0: tt_l // 2], v_cur)
        ]).flatten()
        uv = fsolve(fc, np.hstack((u_cur, v_cur)))
        v_tilde = uv[tt_l // 2:]  # 将位置分量取出并进行投影
        if proj_dim == 1:
            v_cur = simple_newton_sca(manifold, v_tilde)
        uv[tt_l // 2:] = v_cur
        model.add_res(uv)
        u_cur, v_cur = uv[0: tt_l // 2], uv[tt_l // 2:]
    return model.get_res()
