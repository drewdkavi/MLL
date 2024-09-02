import random
import numpy as np

from package.Models.Classifier.SVM import Binary_SVM
from package.data import get_4blobs
from package.Tools import split_test_train, plot_2d_predictions, LabelEncoder
import matplotlib.pyplot as plt


def demo():
    print("Linear kernel support Vector Machine - One vs. One.  On four classes")

    '''
    x2
    | xx / xxxx
    |xx /xxxxxxx
    |x /xx xxxxxx
    | /     xxxx
    |/______________: x1
    '''

    # DATA:

    NUM_DATA_POINTS = 100
    NUM_FEATURES = 2

    xs, ys = get_4blobs(NUM_DATA_POINTS, add_extra_dim=True)

    # Prep. data

    le = LabelEncoder()
    le.build(ys)
    ys_enc = le.fit_transform(ys)
    x_train, x_test, y_train, y_test = split_test_train(xs, ys_enc)

    # Initialise and train model

    svm = Binary_SVM.SVMLinearOVO(input_dim=NUM_FEATURES, num_classes=4)
    svm.train(x_train, y_train)

    # Plot how the SVM classifies feature space

    plot_2d_predictions(svm, x_train, y_train, 0, 1, 0, 1, add_extra_dim=True)

    # Display results of testing the model

    print(f"accuracy: {svm.test(x_test, y_test).accuracy()}")





