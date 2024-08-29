import numpy as np
from package.Models.Neural_Network import SNN
from package.Models.ModelsTemplate import Classifier
import numpy.typing as npt


class SequentialNeuralNetwork(Classifier):

    def __init__(self,
                 input_dimension: int,
                 num_classes: int,
                 *,
                 INTERNAL_LAYERS: list[int] | None = None):
        super().__init__(input_dimension, num_classes)
        layers = INTERNAL_LAYERS + [num_classes] if INTERNAL_LAYERS else [num_classes]

        self.layers: list[
            npt.NDArray[
                npt.NDArray[float]
            ]
        ] = []
        # self.layers is a 3-d array
        # let l represent the l-th layer
        # self.layers(l, i) represents the a size(l-1) vector of the weightings of the i-th unit of the l-th layer
        prev_layer_size = input_dimension
        for index, layer_size in enumerate(layers):
            self.layers.append(
                np.array(
                    [np.random.rand(prev_layer_size) for _ in range(layer_size)]
                )
            )
            prev_layer_size = layer_size
        x_dp = np.array([1.2, 3.22, -0.77, 4.5])
        print(self.predict(x_dp))

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):
        pass

    def predict(self, x_test: npt.NDArray[npt.NDArray[float]]):
        # Categorical Prediction
        return np.argmax(SNN.feedforward(x_test, self.layers))

