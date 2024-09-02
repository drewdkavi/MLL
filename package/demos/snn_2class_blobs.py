import numpy as np
import math
import random

from package.Models.Neural_Network import SequentialNeuralNetworkClassifier
from package.Tools import plot_2d_predictions, LabelEncoder
from package.Tools import split_test_train
from package.Tools import map_categorical, map_decategorical
from package.data import get_2blobs_diagonal


def demo():
    print("Sequential Neural Network Classifier (ReLU & Cross-entropy) - 2 classes ")
    print("_________\n"
          "| X | O |\n"
          "_________\n"
          "| O | X |\n"
          "_________\n"
          "\n")

    #  DATA:

    NUM_DATA_POINTS = 500
    RADIUS = 0.25

    Xs, ys = get_2blobs_diagonal(NUM_DATA_POINTS, RADIUS)

    #  Clean/Process Data
    le = LabelEncoder()
    le.build(ys)
    ys_enc = le.fit_transform(ys)
    x_train, x_test, y_train, y_test = split_test_train(Xs, ys_enc, 0.2)

    y_train_vector = map_categorical(y_train, 2)

    #  Initialise and train model

    nn = SequentialNeuralNetworkClassifier(
        2, 2,
        INTERNAL_LAYERS=[32, 20],
        EPOCHS=2000,
        learning_rate=0.03
    )
    nn.train(x_train, y_train_vector)

    # Plot how the model classifies feature-space

    plot_2d_predictions(nn, x_train, y_train, x1_min=0, x1_max=1, x2_min=0, x2_max=1)

    # Get test results
    result = nn.test(x_test, y_test)

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-macro - {result.f1.average_score()}")
    print(f"F1-micro - {result.f1.global_score()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")
