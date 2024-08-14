from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


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

    @abstractmethod
    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):
        pass

    @abstractmethod
    def predict(self, x_dp: npt.NDArray[float]) -> int | float:
        pass

    # TODO: Implement this
    def test(self, x_test: npt.NDArray[npt.NDArray[float]], y_true: npt.NDArray[any]):
        if len(x_test) != len(y_true):
            raise ValueError(f"Mismatched lengths - {len(x_test)} & {len(y_true)}! for x_test and y_true")
        # print("Testing... ")
        hit_count = 0
        for x_i, y_i in zip(x_test, y_true):
            # print(f"Testing x = {x_i}, against y = {y_i}")
            pred = self.predict(x_i)
            if pred == y_i:
                hit_count += 1
        return hit_count / len(y_true)
