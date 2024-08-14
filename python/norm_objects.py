from python.NormWrapper import NormWrapper
import numpy as np


def square_norm(self, v: float) -> float:
    return v ** 2


def minimiser(self, x_t, y_t):
    x_train_transpose = x_t.T
    XTX_inv = np.linalg.inv(x_train_transpose @ x_t)
    b = XTX_inv @ (x_train_transpose @ y_t)
    return b


square_nw = NormWrapper(square_norm, minimiser)
