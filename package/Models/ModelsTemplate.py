from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

from package.Models.Classifier.ResultObjects import F1Object, Result


class Model(ABC):

    @abstractmethod
    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):
        pass

    @abstractmethod
    def predict(self, x_dp: npt.NDArray[float]) -> int | float:
        pass


class Regressor(Model):
    @abstractmethod
    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):
        pass

    @abstractmethod
    def predict(self, x_dp: npt.NDArray[float]) -> int | float:
        pass


class Classifier(Model):

    def __init__(
            self,
            input_dimension: int,
            num_classes: int):

        if input_dimension < 1:
            raise ValueError(f"input_dimension must be specified: {input_dimension}")
        if num_classes < 1:
            raise ValueError(f"Must require classifying to at least 1 class: num_classes = {num_classes}")
        self._INPUT_DIM: int = input_dimension
        self._num_classes: int = num_classes

    @abstractmethod
    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):
        pass

    @abstractmethod
    def predict(self, x_dp: npt.NDArray[float]) -> int | float:
        pass

    def test(self, x_test: npt.NDArray[npt.NDArray[float]], y_true: npt.NDArray[any]):

        print(f"Overridden testing... ")
        if len(x_test) != len(y_true):
            raise ValueError(f"Mismatched lengths - {len(x_test)} & {len(y_true)}! for x_test and y_true")

        f1s = np.zeros((self._num_classes, 4), dtype=int)
        num_success: int = 0
        num_total: int = 0
        # each element of 'results' is (TP, FP, FN) for each category:
        for x_i, y_i in zip(x_test, y_true):
            # print(f"Testing x = {x_i}, against y = {y_i}")
            pred = self.predict(x_i)
            if y_i == pred:
                num_success += 1
                f1s[pred, 0] += 1
            else:
                # FP for pred category
                f1s[pred, 1] += 1
                # FN for y_i
                f1s[y_i, 2] += 1

            f1s[y_i, 3] += 1
            num_total += 1

        f1_object = F1Object(f1s)
        result = Result(num_total, num_success, f1_object)

        return result
