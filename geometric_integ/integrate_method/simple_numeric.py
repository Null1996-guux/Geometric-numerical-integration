# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:18:17 2020

@author: Math
"""

import numpy as np
from scipy.optimize import fsolve

def explicit_euler(model):
    y_cur = model.y0
    for i in range(model.epoch):
        y_cur = y_cur + model.h * model.f(y_cur)
        model.add_res(y_cur)
    return model.get_res()

def implicit_euler(model):
    y_cur = model.y0
    for i in range(model.epoch):
        fc = lambda x: y_cur + model.h * model.f(x) - x
        y_cur = fsolve(fc, y_cur)
        model.add_res(y_cur)
    return model.get_res()

def implicit_midpoint(model):
    y_cur = model.y0
    for i in range(model.epoch):
        fc = lambda x: y_cur + model.h * model.f( (y_cur + x) / 2 ) - x
        y_cur = fsolve(fc, y_cur)
        model.add_res(y_cur)
    return model.get_res()

def symplectic_euler(model, switch = 0):
    assert switch == 0 or switch == 1, 'switch must be equal to 0 or 1'
    tt_l = len(model.y0)
    u_cur, v_cur = model.y0[0: tt_l // 2], model.y0[tt_l // 2:]
    a = lambda u, v: model.f(np.hstack((u, v)))[0: tt_l // 2]
    b = lambda u, v: model.f(np.hstack((u, v)))[tt_l // 2 :]
    if switch == 0:
        for i in range(model.epoch):
            fc = lambda x: np.array([
                x[0: tt_l // 2]  - u_cur - model.h * a(u_cur, x[tt_l // 2 :]),
                x[tt_l // 2:] - v_cur - model.h * b(u_cur, x[tt_l // 2 :])
            ]).flatten()
            uv = fsolve(fc, np.hstack((u_cur, v_cur)))
            model.add_res(uv)
            u_cur, v_cur = uv[0: tt_l // 2], uv[tt_l // 2:]
        return model.get_res()

def stormer_verlet(model):
    tt_l = len(model.y0)
    p_cur, q_cur = model.y0[0: tt_l // 2], model.y0[tt_l // 2:]
    for i in range(model.epoch):
        p_mid = p_cur + (model.h / 2) * model.f_stormer(q_cur)
        q_cur = q_cur + model.h * p_mid
        p_cur = p_mid + (model.h / 2) * model.f_stormer(q_cur)
        model.add_res(np.hstack((p_cur, q_cur)))
    return model.get_res()