import numpy.typing as npt
import numpy as np
from typing import TypeVar, Tuple

T = TypeVar('T')


def split_test_train(X: npt.NDArray[npt.NDArray[float]], y: npt.NDArray[T], test_size: float = 0.25) -> Tuple[
    npt.NDArray[npt.NDArray[float]],
    npt.NDArray[npt.NDArray[float]],
    npt.NDArray[T],
    npt.NDArray[T]
]:
    """
    Split data into train and test sets.
    :param X:
    :param y:
    :param test_size:
    :return: X_train, X_test, y_train, y_test
    """
    N = len(X)
    if N != len(y):
        raise ValueError(f'X and y must have the same length')
    test_split, train_split = np.array_split(np.random.permutation(np.arange(N)), [int(N * test_size)])

    return X[train_split], X[test_split], y[train_split], y[test_split]
