import numpy.typing as npt
import numpy as np


def get_map(num_categories, to_int: bool = False) -> npt.NDArray[int | float]:
    def f(y: int | float):
        if to_int:
            v: npt.NDArray[int] = np.zeros(num_categories, dtype=int)
            v[y] = 1
            return v
        else:
            v: npt.NDArray[int] = np.zeros(num_categories)
            v[y] = 1.0

            return v
    return f


def map_categorical(ys: npt.NDArray[int | float], num_categories, to_int: bool = False):
    mapping = get_map(num_categories, to_int=to_int)
    return np.array([mapping(y) for y in ys])


def map_decategorical(ys):
    return np.array([np.argmax(y) for y in ys])