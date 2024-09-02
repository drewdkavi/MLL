from typing import List
import numpy.typing as npt
import numpy as np
from package.Models.ModelsTemplate import Classifier
from scipy.optimize import minimize
import cProfile, pstats


# Data MUST!!! be separable to run
class BinarySVMLinearSeparable(Classifier):
    """ Data MUST be separable """

    def __init__(self, input_dim: int):
        if input_dim < 0:
            raise ValueError("Model must have input dimension >= 0")
        self._INPUT_DIM = input_dim
        self._weights = [1] * (self._INPUT_DIM + 1)  # this is our 'b'

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int]):

        print('Training')
        N = len(x_train)

        def obj(w):
            return np.inner(w, w)

        def get_constraint_i(i: int):
            # print(f"Getting constraint i: {i}")

            def fun(w):
                dot = np.dot(w, x_train[i])
                variable_part = y_train[i] * dot
                return variable_part - 1

            return fun

        constraints = [{'type': 'ineq', 'fun': get_constraint_i(i)} for i in range(N)]

        result = minimize(obj, x0=np.array(self._weights), constraints=constraints)
        self._weights = result.x
        return result.x

    def predict(self, x_dp: npt.NDArray[float]):

        v = np.dot(self._weights, x_dp)
        # print(f"x_dp = {x_dp}, v(x_dp) = {v}")
        if v > 0:
            return 1
        else:
            return -1

    def test(self, x_test, y_true):
        hit_count = 0
        for x_i, y_i in zip(x_test, y_true):
            pred = self.predict(x_i)
            if pred == y_i:
                hit_count += 1
        return hit_count / len(y_true)


class BinarySVMLinear(Classifier):
    def __init__(self, input_dim: int):
        if input_dim < 0:
            raise ValueError("Model must have input dimension >= 0")
        self._INPUT_DIM = input_dim
        self._weights = [1] * (self._INPUT_DIM + 1)  # this is our 'b'

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int]):

        profiler = cProfile.Profile()
        profiler.enable()

        print('Training... ')
        N = len(x_train)
        z = [0] * N
        wz = self._weights + z

        def obj(inp_wz):
            w = inp_wz[0:self._INPUT_DIM + 1]
            z = inp_wz[self._INPUT_DIM + 1:]
            return 0.5 * np.inner(w, w) + 1 * sum(z)

        def get_constraint_i(i: int):
            # print(f"Getting constraint i: {i}")

            def fun(wz):
                z = wz[self._INPUT_DIM + 1:]
                w = wz[0:self._INPUT_DIM + 1]
                dot = np.dot(w, x_train[i])
                variable_part = y_train[i] * dot
                return variable_part + z[i] - 1

            return fun

        def get_zi_constraint(i: int):
            def fun(wz):
                z = wz[self._INPUT_DIM + 1:]
                return z[i]

            return fun

        constraints = [{'type': 'ineq', 'fun': get_constraint_i(i)} for i in range(N)] \
                      + [{'type': 'ineq', 'fun': get_zi_constraint(i)} for i in range(self._INPUT_DIM + 1, N)]

        result = minimize(obj, x0=wz, constraints=constraints)
        w = result.x[0:self._INPUT_DIM + 1]
        z = result.x[self._INPUT_DIM + 1:]
        # print(f"z: == {z}")
        self._weights = w

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
        return w

    def predict(self, x_dp: npt.NDArray[float]):

        v = np.dot(self._weights, x_dp)
        # print(f"x_dp = {x_dp}, v(x_dp) = {v}")
        if v > 0:
            return 1
        else:
            return -1

    def test(self, x_test, y_true):
        hit_count = 0
        for x_i, y_i in zip(x_test, y_true):
            pred = self.predict(x_i)
            if pred == y_i:
                hit_count += 1
        return hit_count / len(y_true)


class SVMLinearOVR(Classifier):
    """k-way Linear separator for k way linearly separable data"""

    def __init__(self, *, input_dim: int = 0, num_classes: int = 0):
        self._INPUT_DIM = input_dim
        self._num_classes = num_classes
        self._weights = [[0] * (input_dim + 1)] * num_classes

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int]):
        # OVR: y_train expected to be [0, ...., num-classes-1]
        # for each i in this, encode y to 1 for i and -1 for not i
        # then compute the OVR weighting for it
        # pass it to weight i
        # we are done!
        for i in range(self._num_classes):
            print(f"Training class {i} for O vs. R: ", end="\r")
            self._weights[i] = self._trainOVR(x_train, y_train, i)
        print("                                                      ")
        return self._weights

    def _trainOVR(self, x_train, y_train, i: int):
        _X_TRAIN = x_train
        _Y_TRAIN_OVR_ENCODED = [1 if y == i else -1 for y in y_train]

        N = len(_X_TRAIN)
        z = [0] * N
        wz = self._weights[i] + z

        def obj(inp_wz):
            w = inp_wz[0:self._INPUT_DIM + 1]
            z = inp_wz[self._INPUT_DIM + 1:]
            return 0.5 * np.inner(w, w) + 1000 * sum(z)

        def get_constraint_i(i: int):
            # print(f"Getting constraint i: {i}")

            def fun(wz) -> float:
                z = wz[self._INPUT_DIM + 1:]
                w = wz[0:self._INPUT_DIM + 1]
                dot = np.dot(w, _X_TRAIN[i])
                variable_part = _Y_TRAIN_OVR_ENCODED[i] * dot
                return variable_part + z[i] - 1

            return fun

        def get_zi_constraint(i: int):
            def fun(wz):
                z = wz[self._INPUT_DIM + 1:]
                return z[i]

            return fun

        constraints = [{'type': 'ineq', 'fun': get_constraint_i(i)} for i in range(N)] \
                      + [{'type': 'ineq', 'fun': get_zi_constraint(i)} for i in range(self._INPUT_DIM + 1, N)]

        result = minimize(obj, x0=wz, constraints=constraints)
        w = result.x[0:self._INPUT_DIM + 1]
        z = result.x[self._INPUT_DIM + 1:]
        # print(f"z: == {z}")
        return w

    def predict(self, x_dp: npt.NDArray[float]) -> int | float:

        p_vals: List[float] = [float(np.dot(self._weights[i], x_dp)) for i in range(self._num_classes)]
        classification: int = max(range(len(p_vals)), key=lambda i: p_vals[i])
        return classification

    def test(self, x_test: npt.NDArray[npt.NDArray[float]], y_true: npt.NDArray[int]):
        hit_count = 0
        for x_i, y_i in zip(x_test, y_true):
            pred = self.predict(x_i)
            if pred == y_i:
                hit_count += 1
        return hit_count / len(y_true)


class SVMLinearOVRSeparable(Classifier):
    """k-way Linear separator for k way linearly separable data"""

    def __init__(self, *, input_dim: int = 0, num_classes: int = 0):
        self._INPUT_DIM = input_dim
        self._num_classes = num_classes
        self._weights = [[0] * (input_dim + 1)] * num_classes

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int]):
        # OVR: y_train expected to be [0, ...., num-classes-1]
        # for each i in this, encode y to 1 for i and -1 for not i
        # then compute the OVR weighting for it
        # pass it to weight i
        # we are done!
        for i in range(self._num_classes):
            print(f"Training O vs. R for class {i}: ", end="\r")
            self._weights[i] = self._trainOVO(x_train, y_train, i)
        return self._weights

    def _trainOVR(self, x_train, y_train, i: int):
        _X_TRAIN = x_train
        _Y_TRAIN_OVR_ENCODED = [1 if y == i else -1 for y in y_train]

        print('Training... ')
        N = len(_X_TRAIN)
        w = np.array(self._weights[i])

        def obj(inp_wz) -> float:
            w = inp_wz
            return float(np.inner(w, w))

        def get_constraint_i(i: int):
            # print(f"Getting constraint i: {i}")

            def fun(w) -> float:
                w = w
                dot = np.dot(w, _X_TRAIN[i])
                variable_part = _Y_TRAIN_OVR_ENCODED[i] * dot
                return variable_part - 1

        constraints = [{'type': 'ineq', 'fun': get_constraint_i(i)} for i in range(N)]

        result = minimize(obj, x0=w, constraints=constraints)
        w = result.x
        # print(f"z: == {z}")
        return w

    def test(self, x_test: npt.NDArray[npt.NDArray[float]], y_true: npt.NDArray[int]):
        hit_count = 0
        for x_i, y_i in zip(x_test, y_true):
            pred = self.predict(x_i)
            if pred == y_i:
                hit_count += 1
        return hit_count / len(y_true)

    def predict(self, x_dp: npt.NDArray[float]) -> int | float:

        p_vals: List[float] = [float(np.dot(self._weights[i], x_dp)) for i in range(self._num_classes)]
        classification: int = max(range(len(p_vals)), key=lambda i: p_vals[i])

        return classification


class SVMLinearOVO(Classifier):
    # raise NotImplementedError
    """k-way Linear separator for k way linearly separable data"""

    def __init__(self, *, input_dim: int = 0, num_classes: int = 0):
        self._INPUT_DIM = input_dim
        self._num_classes = num_classes
        # self._weights = [[[0] * (input_dim + 1)] * num_classes] * num_classes
        self._weights = np.zeros((num_classes, num_classes, input_dim + 1))

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int]):
        # OVR: y_train expected to be [0, ...., num-classes-1]
        # for each i in this, encode y to 1 for i and -1 for not i
        # then compute the OVR weighting for it
        # pass it to weight i
        # we are done!
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                if i != j:
                    # print(f"i, j = {i, j}: ", end="\n")
                    self._weights[i][j] = self._trainOVO(x_train, y_train, i, j)
                    # print(f"self._weights[i] = {self._weights[i]}")
        return self._weights

    def _trainOVO(self, x_train, y_train, i: int, j: int):
        """i vs j"""
        print(f"Training {i} vs. {j}", end='\r')
        _X_TRAIN = x_train
        _Y_TRAIN_OVR_ENCODED = [1 if y == i else (-1 if y == j else 0) for y in y_train]
        # print(f"_Y_TRAIN: {_Y_TRAIN_OVR_ENCODED}")
        N = len(_X_TRAIN)
        z = [0] * N
        # print(f'self._weights[i][j] = {self._weights[i][j]}, of shape: {self._weights[i][j].shape}')
        # print(f'z = {z}, of shape: {z.shape}')

        wz = self._weights[i][j].tolist() + z

        def obj(inp_wz):
            w = inp_wz[0:self._INPUT_DIM + 1]
            z = inp_wz[self._INPUT_DIM + 1:]
            return 0.5 * np.inner(w, w) + 10 * sum(z)

        def get_constraint_i(i: int):
            # print(f"Getting constraint i: {i}")

            def fun(wz) -> float:
                z = wz[self._INPUT_DIM + 1:]
                w = wz[0:self._INPUT_DIM + 1]
                dot = np.dot(w, _X_TRAIN[i])
                variable_part = _Y_TRAIN_OVR_ENCODED[i] * dot
                return variable_part + z[i] - 1

            return fun

        def get_zi_constraint(i: int):
            def fun(wz):
                z = wz[self._INPUT_DIM + 1:]
                return z[i]

            return fun

        constraints = [{'type': 'ineq', 'fun': get_constraint_i(i)} for i in range(N)] \
                      + [{'type': 'ineq', 'fun': get_zi_constraint(i)} for i in range(self._INPUT_DIM + 1, N)]

        result = minimize(obj, x0=wz, constraints=constraints)
        w = result.x[0:self._INPUT_DIM + 1]
        z = result.x[self._INPUT_DIM + 1:]
        # print(f"z: == {z}")
        return w

    def predict(self, x_dp: npt.NDArray[float]) -> int | float:
        max_i = 0
        max_v = 0
        for i in range(self._num_classes):

            pred_i_against = [np.dot(wij, x_dp) for wij in self._weights[i]]
            t = sum(pred_i_against)
            if t > max_v:
                max_v = t
                max_i = i
        return max_i

    def test(self, x_test: npt.NDArray[npt.NDArray[float]], y_true: npt.NDArray[int]):
        print('Testing... ')
        hit_count = 0
        for x_i, y_i in zip(x_test, y_true):
            pred = self.predict(x_i)
            if pred == y_i:
                hit_count += 1
        return hit_count / len(y_true)
