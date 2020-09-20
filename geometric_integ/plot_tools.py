from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# use this function to draw 2d contour conveniently
def my_draw_contour2d(func, var1_devid, var2_devid, curve_number = 10, is_ilustrate = True):
    x = np.linspace(var1_devid[0],  var1_devid[1],  var1_devid[2])
    y = np.linspace(var2_devid[0], var2_devid[1],  var2_devid[2])
    X, Y = np.meshgrid(x, y); Z = func(X, Y)
    ctor = plt.contour(X, Y, Z, curve_number)
    if is_ilustrate:
        plt.clabel(ctor)

# plot res record of numeric solutio in phase space
def draw_numeric2d(res_record, color = "red", marker = None, label = None):
    x_plt = []; y_plt = []
    for point in res_record:
        x_plt.append(point[0]); y_plt.append(point[1])
    plt.plot(x_plt, y_plt, color = color, marker = marker, label = label)

def hamiltonian_energy_loss2d(model, res_record, color = 'blue', label = None):
    tmesh = model.time_mesh()
    rec = np.zeros(len(tmesh))
    for i in range(len(rec)):
        rec[i] = model.H(res_record[i]) - model.H0
    plt.plot(tmesh, rec, color = color, label = label)
