from package.Models.Classifier.Random_Forest import RandomForest
from package.Tools import split_test_train
from package.Tools import map_categorical, map_decategorical
from sklearn.datasets import load_iris


def demo():
    print("Random-Forest Model on the Iris Data Set")

    #  Fetch scikit-learns data

    xs, ys = load_iris(return_X_y=True)

    #  Clean/Process Data:

    x_train, x_test, y_train, y_test = split_test_train(xs, ys, 0.2)

    #  Initialise and train model

    nn = RandomForest.RandomForest(
        4, 3,
        NUM_ESTIMATORS=100,
        COL_SUBSAMPLE_SIZE=4
    )
    nn.train(x_train, y_train)

    # Get test results
    result = nn.test(x_test, map_decategorical(y_test))

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-macro - {result.f1.average_score()}")
    print(f"F1-micro - {result.f1.global_score()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")

