import numpy.typing as npt
import numpy as np
from typing import TypeVar, Dict, Tuple, Any

T = TypeVar('T')


class LabelEncoder:

    def __init__(self):
        self.encode = {}
        self.decode = {}
        self.get_enc = None
        self.get_dec = None
        self.N: int = 0

    def build(self, y_classes: npt.NDArray[T]) -> None:
        unique = np.unique(y_classes)
        enum_unique = enumerate(unique)
        encode: Dict[T, int] = {label: int_id for int_id, label in enum_unique}
        decode: Dict[int, T] = {int_id: label for int_id, label in enum_unique}
        self.encode, self.decode = encode, decode
        N = len(self.encode)
        self.N = N

        def _get_encoded(label: T) -> int:
            return self.encode[label]

        def _get_decoded(int_id: int) -> T:
            return self.decode[int_id]

        get_encoded = np.vectorize(_get_encoded)
        get_decoded = np.vectorize(_get_decoded)
        self.get_enc = get_encoded
        self.get_dec = get_decoded

    def fit_transform(self, y_classes: npt.NDArray[T]) -> npt.NDArray[int]:
        return self.get_enc(y_classes)

    def inv_transform(self, y_classes: npt.NDArray[int]) -> npt.NDArray[T]:
        return self.get_dec(y_classes)

    def get_num_classes(self) -> int:
        return self.N
