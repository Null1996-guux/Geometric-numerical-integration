import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

'''
RK : 

A : matrix A of butcher tab
b : matrix b of butcher tab

'''
class RK:
    def __init__(self, A, b, ndim = 1):
        self.A = A
        self.b = b.reshape(-1, 1)
        self.c = np.sum(A, axis = 1).reshape(-1, 1)
        self.stage = len(b)
        self.ndim = ndim
        
    def __fhelper(self, model, Y):
        YY = Y.flatten()
        for i in range(self.stage):
            YY[i * self.ndim : (i + 1) * self.ndim] = model.f(YY[i * self.ndim : (i + 1) * self.ndim])
        return YY.reshape(-1, 1)
            
    def odesolve(self, model):
        e = np.ones(self.stage, dtype = np.float64).reshape(-1, 1)
        Id = np.eye(self.ndim, dtype = np.float64)
        y = model.y0.reshape(-1, 1); h = model.h
        for i in range(model.epoch):
            ff = lambda Y: (np.kron(e, y) + h * np.dot(np.kron(self.A, Id), self.__fhelper(model, Y)) - Y.reshape(-1, 1)).flatten()
            Ysol = fsolve(ff, np.kron(e, model.f(model.y0).reshape(-1, 1))).reshape(-1, 1)
            y = y + h * np.dot(np.kron(self.b.T, Id), self.__fhelper(model, Ysol))
            model.add_res(y.flatten())
        return model.res   

# Then we will test program on the well known Lotka model by RK3, RK4, Gauss4

'''
Data of Lotka Model
'''
class DataModel:
    def __init__(self, t0, y0, h, epoch, data_range = 'all'):
        self.t0 = t0
        self.y0 = y0
        self.h = h
        self.epoch = epoch
        self.data_range = data_range
        if self.data_range == 'all':
            self.res = [self.y0]
        else:
            self.res = [self.y0[data_range[0]: data_range[1]]]

    def time_mesh(self):
        end = self.t0 + self.epoch * self.h
        t = np.linspace(self.t0, end, self.epoch + 1)
        return t

    def add_res(self, y):
        self.res.append(y) if self.data_range == 'all' \
            else self.res.append(y[self.data_range[0] : self.data_range[1]])

    def get_res(self):
        return self.res
    
class LotkaModel(DataModel):
    def __init__(self, t0, y0, h, epoch, data_range = 'all'):
        DataModel.__init__(self, t0, y0, h, epoch, data_range)

    def f(self, y):
        return np.array([y[0] * (y[1] - 2), y[1] * (1 - y[0])])

'''
visuable function
'''
def my_draw_contour2d(func, var1_devid, var2_devid, curve_number = 10, is_ilustrate = True):
    x = np.linspace(var1_devid[0],  var1_devid[1],  var1_devid[2])
    y = np.linspace(var2_devid[0], var2_devid[1],  var2_devid[2])
    X, Y = np.meshgrid(x, y); Z = func(X, Y)
    ctor = plt.contour(X, Y, Z, curve_number)
    if is_ilustrate:
        plt.clabel(ctor)
        
def draw_numeric2d(res_record, color = "red", marker = None, label = None):
    x_plt = []; y_plt = []
    for point in res_record:
        x_plt.append(point[0]); y_plt.append(point[1])
    plt.plot(x_plt, y_plt, color = color, marker = marker, label = label)
    
'''
test function

'''
def RK3(model, ndim):
    A = np.array([ [0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0] ], dtype = np.float64)
    b = np.array([1 / 6, 2 / 3, 1 / 6], dtype = np.float64)
    rk = RK(A, b, ndim)
    res = rk.odesolve(model)
    return res

def RK4(model, ndim):
    A = np.array([ [0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0] ], dtype = np.float64)
    b = np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6], dtype = np.float64)
    rk = RK(A, b, ndim)
    res = rk.odesolve(model)
    return res

def Gauss4(model, ndim):
    A = np.array([ [1 / 4, 1 / 4 - np.sqrt(3) / 6], [1 / 4 + np.sqrt(3) / 6, 1 / 4] ], dtype = np.float64)
    b = np.array([1 / 2, 1 / 2], dtype = np.float64)
    rk = RK(A, b, ndim)
    res = rk.odesolve(model)
    return res

def RK_test():
    y0 = np.array([4, 8], dtype = np.float64)
    h = [0.2, 0.2, 0.2]
    epoch = [100, 100, 100]
    ndim = 2
    
    methods = [RK3, RK4, Gauss4]
    labels = ['RK3 res', 'RK4 res', 'Gauss4 res']

    f_lotka = lambda u, v: np.log(u) - u + 2 * np.log(v) - v
    

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        model = LotkaModel(0, y0, h[i], epoch[i])
        res = methods[i](model, ndim)
        my_draw_contour2d(f_lotka, (0.01, 8, 100), (0.01, 12, 100), curve_number = 20)
        draw_numeric2d(res, label= labels[i])
        plt.legend()
    plt.show()
 
if __name__ == '__main__':
    RK_test()
