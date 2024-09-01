import random

import numpy as np
import cProfile, pstats
from package.Models.Neural_Network import SNN2
from package.Models.ModelsTemplate import Classifier
import numpy.typing as npt


class SequentialNeuralNetworkClassifier(Classifier):

    def __init__(self,
                 input_dimension: int,
                 num_classes: int,
                 *,
                 EPOCHS: int = 10,
                 learning_rate: float = 0.01,
                 INTERNAL_LAYERS: list[int] | None = None):
        super().__init__(input_dimension, num_classes)
        layers = INTERNAL_LAYERS + [num_classes] if INTERNAL_LAYERS else [num_classes]
        self.epochs = EPOCHS
        self.lr = learning_rate

        self.layers: list[
            npt.NDArray[
                npt.NDArray[float]
            ]
        ] = []
        # self.layers is a 3-d array
        # let l represent the l-th layer
        # self.layers(l, i) represents the a size(l-1) vector of the weightings of the i-th unit of the l-th layer
        self.biases: list[
            npt.NDArray[float]
        ] = []
        prev_layer_size = input_dimension
        for index, layer_size in enumerate(layers):
            self.layers.append(
                np.array(
                    [np.random.uniform(-1, 1, size=prev_layer_size) for _ in range(layer_size)]
                )
            )
            self.biases.append(
                np.random.uniform(-0.05, 0.05, size=layer_size)
            )
            prev_layer_size = layer_size

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):

        # p1 = cProfile.Profile()
        # p1.enable()


        def update_layers(deltas: list, num):
            for index_layer, layer in enumerate(self.layers):
                self.layers[index_layer] = layer - self.lr * 1/num * deltas[index_layer]

        def update_biases(deltas: list, num):
            for index_layer, layer in enumerate(self.biases):
                self.biases[index_layer] = layer - self.lr * 1/num * deltas[index_layer]

        def update_weights(deltas: list, weights: list):
            for index_layer, layer in enumerate(weights):
                weights[index_layer] = layer + deltas[index_layer]
            return weights

        N = len(x_train)
        T = self.epochs  # number of training cycles
        weights = None
        biases = None

        for training_cycle in range(T):
            print(f"Training cycle - {training_cycle + 1}/{T}", end='\r')
            NUM_STOCHASTIC = 20
            for j in range(NUM_STOCHASTIC):
                # stochastic 1 datapoint simple gradient descent
                dp_index = random.randint(0, N - 1)
                xdp, ydp = x_train[dp_index], y_train[dp_index]

                activations, predicted = SNN2.feedforward_all(xdp, self.layers, self.biases)
                weight_deltas, bias_deltas = SNN2._bp_single(
                    self.layers,
                    activations,
                    ydp,
                    xdp,
                    predicted
                )
                if j == 0:
                    weights = weight_deltas
                    biases = bias_deltas
                else:
                    weights = update_weights(weights, weight_deltas)
                    biases = update_weights(biases, bias_deltas)
            update_layers(weights, NUM_STOCHASTIC)
            update_biases(biases, NUM_STOCHASTIC)

        # p1.disable()
        # stats = pstats.Stats(p1).sort_stats('cumtime')
        # stats.print_stats()

    def ff(self, x_test):
        return SNN2.feedforward(x_test, self.layers, self.biases)

    def predict(self, x_test: npt.NDArray[npt.NDArray[float]]):
        # Categorical Prediction
        return np.argmax(SNN2.feedforward(x_test, self.layers, self.biases))
