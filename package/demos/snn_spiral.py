import cProfile, pstats

from package.Models.Neural_Network import SequentialNeuralNetworkClassifier
from package.Tools import split_test_train, plot_2d_predictions
from package.Tools.Extras import map_extra_2d_features
from package.Tools import map_categorical
from package.data import get_twin_spiral_data


def demo():
    print("SNN classifying twin spirals")

    # DATA:

    NUM_DATA_POINTS = 250

    Xs, ys = get_twin_spiral_data(300)

    # Clean/Process data
    Xs_extras = map_extra_2d_features(Xs)  # maps [x1, x2] -> [x1, x2, x1^2, x2^2, sin(x1), sin(x2)]
    x_train, x_test, y_train, y_test = split_test_train(Xs_extras, ys, 0.3)

    ys_train_vectored = map_categorical(y_train, 2)

    #  Initialise and train model

    nn = SequentialNeuralNetworkClassifier(
        6, 2,
        learning_rate=0.1,
        EPOCHS=2500, INTERNAL_LAYERS=[10, 10]
    )

    # Uncomment the following to profile the model
    # requires cProfile, & pStats

    # p1 = cProfile.Profile()
    # p1.enable()
    nn.train(x_train, ys_train_vectored)
    # p1.disable()
    # stats = pstats.Stats(p1).sort_stats('cumtime')
    # stats.print_stats()

    # Plot how the model classifies feature-space

    plot_2d_predictions(nn, x_train, y_train, -3.1, 3.1, -3.1, 3.1, extras=True)

    # Get test results
    result = nn.test(x_test, y_test)

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")



