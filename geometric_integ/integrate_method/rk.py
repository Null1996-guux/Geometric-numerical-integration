import numpy as np
from scipy.optimize import fsolve

class RK:
    def __init__(self, model, stage, butcher):
        self.model = model
        self.stage = stage
        self.butcher = butcher