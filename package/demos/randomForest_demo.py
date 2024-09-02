import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from package.Models.Classifier.Random_Forest import RandomForest
from package.data import get_4blobs
from package.Tools import LabelEncoder, split_test_train, plot_2d_predictions
import cProfile, pstats


def demo():
    print("Random Forest Demo - 100k data points, 30 estimator decision trees")
    #  DATA:

    NUM_DATA_POINTS = 100_000
    EPSILON = 0.25

    Xs, ys = get_4blobs(NUM_DATA_POINTS, EPSILON)

    #  Clean/Process Data:
    label_enc = LabelEncoder()
    label_enc.build(ys)
    enc_ys = label_enc.fit_transform(ys)
    x_train, x_test, y_train, y_test = split_test_train(Xs, enc_ys, 0.2)

    #  Initialise and train model

    rf = RandomForest.RandomForest(2, 4, NUM_ESTIMATORS=30, COL_SUBSAMPLE_SIZE=4)

    p1 = cProfile.Profile()
    p1.enable()

    rf.train(x_train, y_train)

    p1.disable()
    stats = pstats.Stats(p1).sort_stats('cumtime')
    stats.print_stats()

    # Plot how the model classifies feature-space

    plot_2d_predictions(rf, x_train, y_train, 0, 1, 0, 1)

    # Get test results and compare with sklearn's own model

    #       sklearn's model:

    # p2 = cProfile.Profile()
    # p2.enable()
    model = RandomForestClassifier(n_estimators=30)
    model.fit(x_train, y_train)
    # p2.disable()
    # stats = pstats.Stats(p2).sort_stats('cumtime')
    # stats.print_stats()
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy - sklearn: {accuracy}")

    #       This package's model:

    result = rf.test(x_test, y_test)

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-macro - {result.f1.average_score()}")
    print(f"F1-micro - {result.f1.global_score()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")
    print(f"F1-print - {result.f1}")
