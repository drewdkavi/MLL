import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

from package.Models.Neural_Network import SequentialNeuralNetworkClassifier, SequentialNeuralNetworkRegressor
from package.Tools import split_test_train
from package.Tools import map_categorical, map_decategorical
from sklearn.datasets import load_iris, load_breast_cancer


def demo():
    print("Sequential Neural Network on the Breast Cancer Data Set")


    #  Fetch scikit-learns data

    xs, ys = load_breast_cancer(return_X_y=True)

    #  Clean/Process Data:

    ohe_ys = map_categorical(ys, 2)
    x_train, x_test, y_train, y_test = split_test_train(xs, ohe_ys, 0.5)

    #  Initialise and train model
    #  -> high dimensionality so take a low learning rate

    nn = SequentialNeuralNetworkClassifier(
        30, 2,
        INTERNAL_LAYERS=[64, 64],
        EPOCHS=2000, learning_rate=0.0001
    )
    nn.train(x_train, y_train)

    # Get test results
    result = nn.test(x_test, map_decategorical(y_test))

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-macro - {result.f1.average_score()}")
    print(f"F1-micro - {result.f1.global_score()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")

