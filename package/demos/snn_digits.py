from package.Models.Neural_Network import SequentialNeuralNetworkClassifier
from package.Tools import split_test_train
from package.Tools import map_categorical, map_decategorical
from sklearn.datasets import load_digits


def demo():
    print("SNN on the handwritten digit classification Data Set")

    #  Fetch scikit-learns data

    xs, ys = load_digits(return_X_y=True)

    #  Clean/Process Data:

    ohe_ys = map_categorical(ys, 10)
    x_train, x_test, y_train, y_test = split_test_train(xs, ohe_ys, 0.4)

    #  Initialise and train model
    #  -> very low learning rate, since high dimensionality

    nn = SequentialNeuralNetworkClassifier(
        64, 10,
        INTERNAL_LAYERS=[128, 128, 128],
        EPOCHS=4000, learning_rate=0.0001
    )
    nn.train(x_train, y_train)

    # Get test results
    result = nn.test(x_test, map_decategorical(y_test))

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-macro - {result.f1.average_score()}")
    print(f"F1-micro - {result.f1.global_score()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")

