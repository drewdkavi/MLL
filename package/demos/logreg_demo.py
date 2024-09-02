import random
import numpy as np

from package.Models.Classifier.LogisticClassification import LogisticReg
from package.Tools import split_test_train, plot_2d_predictions


def demo():
    print("Binary logistic regression - on a linearly separable dataset")

    # DATA

    NUM_DATA_POINTS = 400
    NUM_FEATURES = 2

    xs = [[random.uniform(0, 1) for _ in range(NUM_FEATURES)] for _ in range(NUM_DATA_POINTS)]
    ys = []
    for x in xs:
        if sum(x) < NUM_FEATURES / 2:
            ys.append(1)
        else:
            ys.append(0)

    # Cleaning/Preprocessing data

    xs = [x + [1] for x in xs]  # for the constant term

    xs = np.array(xs)
    ys = np.array(ys)

    x_train, x_test, y_train, y_test = split_test_train(xs, ys)

    # Initialise and train model

    blr = LogisticReg.BinaryLogisticRegression(NUM_FEATURES)

    blr.train(x_train, y_train)

    # Plot how the model classifies feature space

    plot_2d_predictions(blr,
                        x_train,
                        y_train, 0, 1, 0, 1, add_extra_dim=True)

    # Test Model

    print(f"Accuracy - : {blr.test(x_test, y_test)}")
