
from Classifier.DecisionTrees import DecisionTreeCython
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import cProfile, pstats


def demo():
    NUM_DATA_POINTS = 10000
    NUM_FEATURES = 2
    EPSILON = 0.2

    N = NUM_DATA_POINTS
    xs = []  # : npt.NDArray[npt.NDArray[float]]
    ys = []  # : npt.NDArray[int]
    x1s = []
    x2s = []

    for _ in range(N):

        y = random.choice([0, 1, 2, 3])
        if y == 0:
            ys.append(0)
            x1_init = 0.25
            x2_init = 0.75

        elif y == 1:
            ys.append(1)
            x1_init = 0.45
            x2_init = 0.25

        elif y == 2:
            ys.append(2)
            x1_init = 0.75
            x2_init = 0.45

        elif y == 3:
            ys.append(3)
            x1_init = 0.75
            x2_init = 0.8

        x1 = x1_init + random.uniform(-EPSILON, EPSILON)
        x2 = x2_init + random.uniform(-EPSILON, EPSILON)

        x1s.append(x1)
        x2s.append(x2)
        xs.append(np.array([x1, x2]))
    xs = np.array(xs)
    ys = np.array(ys)
    perm = np.random.permutation(len(xs))
    xs = xs[perm]
    ys = ys[perm]

    x_test = []
    y_test = []
    for _ in range(100):

        y = random.choice([0, 1, 2, 3])
        if y == 0:
            y_test.append(0)
            x1_init = 0.25
            x2_init = 0.75

        elif y == 1:
            y_test.append(1)
            x1_init = 0.45
            x2_init = 0.25

        elif y == 2:
            y_test.append(2)
            x1_init = 0.75
            x2_init = 0.45

        elif y == 3:
            y_test.append(3)
            x1_init = 0.75
            x2_init = 0.8

        x1 = x1_init + random.uniform(-EPSILON, EPSILON)
        x2 = x2_init + random.uniform(-EPSILON, EPSILON)

        x_test.append([x1, x2])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # p1 = cProfile.Profile()
    # p1.enable()
    # dt5 = DecisionTree5.DecisionTree(input_dimension=2, num_classes=4, classes={0, 1, 2, 3})
    # dt5.train(xs, ys)
    # print(f"    Accuracy - DT5: {dt5.test(x_test, y_test)}")
    # p1.disable()
    # stats = pstats.Stats(p1).sort_stats('cumtime')
    # stats.print_stats()

    p2 = cProfile.Profile()
    p2.enable()
    model = DecisionTreeClassifier()
    model.fit(xs, ys)
    print(f"sklearn-tree-depth: {model.tree_.max_depth}")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy - sklearn: {accuracy}")
    p2.disable()
    stats = pstats.Stats(p2).sort_stats('cumtime')
    stats.print_stats()

    # p1 = cProfile.Profile()
    # p1.enable()
    # dt4 = DecisionTree4_new.DecisionTree(input_dimension=2, num_classes=4, classes={0, 1, 2, 3})
    # dt4.train(xs, ys)
    # print(f"    Accuracy - DT4-Random: {dt4.test(x_test, y_test)}")
    # p1.disable()
    # stats = pstats.Stats(p1).sort_stats('cumtime')
    # stats.print_stats()
    #
    # p1 = cProfile.Profile()
    # p1.enable()
    # dt7 = DecisionTree7.DecisionTree(input_dimension=2, num_classes=4,)
    # dt7.train(xs, ys)
    # print(f"    Accuracy - DT7: {dt7.test(x_test, y_test)}")
    # p1.disable()
    # stats = pstats.Stats(p1).sort_stats('cumtime')
    # stats.print_stats()

    p1 = cProfile.Profile()
    p1.enable()
    dtc = DecisionTreeCython.DecisionTree(input_dimension=2, num_classes=4)
    dtc.train(xs, ys)
    print(f"    Accuracy - DT7-Cythonized: {dtc.test(x_test, y_test)}")
    p1.disable()
    stats = pstats.Stats(p1).sort_stats('cumtime')
    stats.print_stats()
