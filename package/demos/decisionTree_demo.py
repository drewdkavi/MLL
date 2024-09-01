
from package.Models.Classifier.DecisionTrees import DecisionTreeCython
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from package.Tools import split_test_train, plot_2d_predictions, LabelEncoder
from package.data import get_4blobs
import cProfile, pstats


def demo():

    #  DATA:

    NUM_DATA_POINTS = 1_000
    EPSILON = 0.3

    Xs, ys = get_4blobs(NUM_DATA_POINTS, EPSILON)

    #  Clean/Process Data:
    le = LabelEncoder()
    le.build(ys)
    ys_enc = le.fit_transform(ys)
    x_train, x_test, y_train, y_test = split_test_train(Xs, ys_enc, 0.3)

    #  Initialise, train and test the two models - sklearn & the package's own model for comparison

    # p1 = cProfile.Profile()
    # p1.enable()
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    print(f"sklearn-tree-depth: {model.tree_.max_depth}")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy - sklearn: {accuracy}")
    # p1.disable()
    # stats = pstats.Stats(p1).sort_stats('cumtime')
    # stats.print_stats()

    # p1 = cProfile.Profile()
    # p1.enable()
    dtc = DecisionTreeCython.DecisionTree(input_dimension=2, num_classes=4)
    dtc.train(x_train, y_train)
    plot_2d_predictions(dtc, x_train, y_train, 0, 1, 0, 1)
    results = dtc.test(x_test, y_test)
    print(f"    Accuracy - This-Model: {results.accuracy()}")
    # p1.disable()
    # stats = pstats.Stats(p1).sort_stats('cumtime')
    # stats.print_stats()
