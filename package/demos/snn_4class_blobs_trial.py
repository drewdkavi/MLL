from package.Models.Neural_Network import SequentialNeuralNetworkClassifier
from package.Tools import LabelEncoder
from package.Tools import split_test_train
from package.Tools import plot_2d_predictions
from package.Tools import map_categorical, map_decategorical
from package.data import get_4blobs


def demo():
    print("Sequential Neural Network Classifier (ReLU & Cross-entropy) - 4 classes ")
    print("_________\n"
          "| S | G |\n"
          "_________\n"
          "| A | B |\n"
          "________ \n"
          "")

    #  DATA:

    NUM_DATA_POINTS = 1_000
    EPSILON = 0.25

    Xs, ys = get_4blobs(NUM_DATA_POINTS, EPSILON)

    #  Clean/Process Data:
    le = LabelEncoder()
    le.build(ys)
    ys_enc = le.fit_transform(ys)
    x_train, x_test, y_train, y_test = split_test_train(Xs, ys_enc, 0.2)

    ys_train_vector = map_categorical(y_train, 4)  # e.g. maps 2 -> [0, 0, 1, 0]

    #  Initialise and train model

    nn = SequentialNeuralNetworkClassifier(
        2, 4,
        INTERNAL_LAYERS=[10, 10], EPOCHS=350,
        learning_rate=0.15
    )
    nn.train(x_train, ys_train_vector)

    # Plot how the model classifies feature-space

    plot_2d_predictions(nn, x_test, y_test, 0, 1, 0, 1)

    # Get test results
    result = nn.test(x_test, y_test)

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-macro - {result.f1.average_score()}")
    print(f"F1-micro - {result.f1.global_score()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")
