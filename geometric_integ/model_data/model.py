import numpy as np

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

