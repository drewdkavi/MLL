from typing import List
import numpy as np
from python.ModelsTemplate import Regressor

'''
Given a data set {y_i, x_i_1, ... , x_i_p} for i = 1 to n, where each y_i, x_i_j are of type float
we get y_i = b . x_i
y_pred = b . x_observation

loss function L(b) = 1/2N * sum( x_i.b - y_i )^2
|-> where each x_i, y_i are the training data
giving grad w.r.t. b of L(b) = 1/N * ((X.T*X)b - X.T*y)
solving for stationary point of L(b) i.e. when loss is minimised gives b = (X.T*X)^-1 * (X.T*y)

'''


def least_squares_minimiser(x_t, y_t):
    x_train_transpose = x_t.T
    XTX_inv = np.linalg.inv(x_train_transpose @ x_t)
    b = XTX_inv @ (x_train_transpose @ y_t)
    return b


class LinearLeastSquareRegressionModel(Regressor):
    def __init__(self, input_dim: int):
        if input_dim < 0:
            raise ValueError("Model must have input dimension >= 0")
        self._INPUT_DIM = input_dim
        self._MODEL_INPUT_DIM = input_dim + 1
        self._weights = [0] * self._MODEL_INPUT_DIM  # this is our 'b'

    def train(self, x_t: List[List[float]], y_t: List[float]):

        x_tt = [x_i + [1] for x_i in x_t]

        x_train = np.array([np.array(x_i) for x_i in x_tt])
        y_train = np.array(y_t)

        self._weights = least_squares_minimiser(x_train, y_train)
        print(self._weights)

    def predict(self, x_dp: List[float]):
        return np.dot(np.array(x_dp), self._weights)

    def _get_weight_i(self, index: int):
        if index < 0 or index > self._INPUT_DIM - 1:
            raise IndexError("Index out of bounds")
        return self._weight[index]

    def _get_intercept(self):
        return self._weight[-1]
