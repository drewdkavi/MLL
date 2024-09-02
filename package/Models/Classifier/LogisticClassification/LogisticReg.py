from package.Models.ModelsTemplate import Classifier
from scipy.optimize import minimize
import numpy as np
import numpy.typing as npt
import numba as nb

'''
y takes values in {0, 1}
'''

DISCRIMINATION_CUTOFF: float = 1 / 2
TRAINING_ITERATIONS = 20
EPSILON = 0.01


@nb.vectorize(["float64(float64)", "float32(float32)"])
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


# def softmax(t: float):
#     raise NotImplementedError


'''
Binary Logistic regression:
Works as follows:
y in {0, 1}, 
We first consider some random variable y', with image {0, 1}

The model has:

P[y' = 1 | x, w] = sigmoid(w * x)
and then we classify by:
predicting y = 1 iff P[y' = 1 | x, w] = sigmoid(w * x) >= 1/2
           y = 0 otherwise.

 
 
'''


class BinaryLogisticRegression(Classifier):

    def __init__(self, input_dim: int):
        if input_dim < 0:
            raise ValueError("Model must have input dimension >= 0")
        self._INPUT_DIM = input_dim
        self._MODEL_INPUT_DIM = input_dim + 1
        self._weights = [0.5] * self._MODEL_INPUT_DIM  # this is our 'b'

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int]):

        N = len(x_train)
        w = self._weights
        S_t = np.zeros((N, N))
        np.fill_diagonal(S_t, 1)

        def make_objective(mu_obj, z_obj):

            def obj(w) -> float:
                def dot_w(x_i):
                    return np.dot(w, x_i)

                obj_sum = 0
                for i in range(N):
                    obj_sum += mu_obj[i] * (1 - mu_obj[i]) * ((z_obj[i] - dot_w(x_train[i])) ** 2)

                return obj_sum

            return obj

        def get_new_mu(x_i: npt.NDArray[float]) -> float:
            t = float(np.dot(w, x_i))
            v = sigmoid(t)
            if v < EPSILON / 2 or 1 - v < EPSILON / 2:
                return EPSILON
            else:
                return v

        # profiler = cProfile.Profile()
        # profiler.enable()
        for t in range(TRAINING_ITERATIONS):
            # print(f"w: {w}")
            mu_t = [get_new_mu(x_i) for x_i in x_train]
            # print(f"Printing mu_t: ", mu_t)
            log_mu = [m * (1 - m) for m in mu_t]

            np.fill_diagonal(S_t, log_mu)  # = S_t // simply updating S_T
            # print(f"Inverted S_t: \n {np.linalg.inv(S_t)}")
            # print(f" np.linalg.inv(S_t) @ (y_train - mu_t): \n {np.linalg.inv(S_t) @ (y_train - mu_t)}")
            z_t = x_train @ w + np.linalg.inv(S_t) @ (y_train - mu_t)
            new_w = minimize(make_objective(mu_t, z_t), w).x
            w = new_w
        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats()
        self._weights = w

    def predict(self, x_dp: npt.NDArray[float]) -> int:

        if np.dot(x_dp, self._weights) > 1 / 2:
            return 1
        else:
            return 0

    def test(self, x_test: npt.NDArray[npt.NDArray[float]], y_true: npt.NDArray[int]):

        hit_count = 0
        for x_i, y_i in zip(x_test, y_true):
            if self.predict(x_i) == y_i:
                hit_count += 1
        return hit_count / len(y_true)
