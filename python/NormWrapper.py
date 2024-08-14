class NormWrapper:
    def __init__(self, norm, get_minimisers):
        """Wrapper containing two things:
        1. A norm 'norm', of type: float -> float
            |-> we implicitly assume 'norm' is a norm; i.e. it follows the axioms of a norm
                that is 1) norm(x) >= 0 & norm(x) == 0 <=> x == 0
                        2) norm(kx) = |k|norm(x)
                        3) norm(x + y) <= norm(x) + norm(y)
        2. A function that takes a matrix X: np.arr[np.arr[float]] and a vector y: np.arr[float]
            where X has dimensions n x p and the vector y has dimensions n x 1
            and return a vector of dimension  p x 1
        """
        self.norm = norm
        self.get_minimisers = get_minimisers


