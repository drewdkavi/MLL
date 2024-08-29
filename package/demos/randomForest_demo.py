import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from package.Models.Classifier.Random_Forest import RandomForest
import matplotlib.pyplot as plt
from package.Tools import LabelEncoder, split_test_train
import numpy as np
import random
import cProfile, pstats


def demo():
    NUM_DATA_POINTS = 10_000
    EPSILON = 0.25
    R = EPSILON
    PI = 3.141592653589793
    N = NUM_DATA_POINTS
    xs = []  # : npt.NDArray[npt.NDArray[float]]
    ys = []  # : npt.NDArray[int]
    x1s = []
    x2s = []
    colours = []
    to_colour = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}
    to_light_colour = {0: 'pink', 1: 'teal', 2: 'lightblue', 3: 'violet'}

    for _ in range(N):

        y = random.choice([0, 1, 2, 3])
        if y == 0:
            ys.append('Apples')
            colours.append(to_colour[0])
            x1_init = 0.25
            x2_init = 0.75

        elif y == 1:
            ys.append('Oranges')
            colours.append(to_colour[1])
            x1_init = 0.35
            x2_init = 0.30

        elif y == 2:
            ys.append('Bananas')
            colours.append(to_colour[2])
            x1_init = 0.75
            x2_init = 0.45

        elif y == 3:
            ys.append('Peaches')
            colours.append(to_colour[3])
            x1_init = 0.75
            x2_init = 0.8
        else:
            x1_init = -1
            x2_init = -1

        r = R * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 1) * 2 * PI
        x1 = x1_init + r * math.cos(theta)
        x2 = x2_init + r * math.sin(theta)

        x1s.append(x1)
        x2s.append(x2)
        xs.append(np.array([x1, x2]))

    label_enc = LabelEncoder()
    xs = np.array(xs)
    ys = np.array(ys)
    label_enc.create(ys)
    enc_ys = label_enc.fit_transform(ys)

    x_train, x_test, y_train, y_test = split_test_train(xs, enc_ys, 0.2)

    rf = RandomForest.RandomForest(2, 4, NUM_ESTIMATORS=200, COL_SUBSAMPLE_SIZE=4)

    p1 = cProfile.Profile()
    p1.enable()
    rf.train(x_train, y_train)
    p1.disable()
    stats = pstats.Stats(p1).sort_stats('cumtime')
    stats.print_stats()

    fig, ax = plt.subplots()

    # mesh_x1s = []
    # mesh_x2s = []
    # mesh_colours = []
    # M = 250
    # for i in range(M):
    #     for j in range(M):
    #         x1ij = i / M
    #         x2ij = j / M
    #         point = [x1ij, x2ij]
    #         mesh_x1s.append(x1ij)
    #         mesh_x2s.append(x2ij)
    #         mesh_colours.append(to_light_colour[rf.predict(np.array(point))])
    #
    # ax.scatter(mesh_x1s, mesh_x2s, c=mesh_colours, s=0.15)

    ax.scatter(x1s, x2s, c=colours, edgecolor='black')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    def onclick(event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:  # Check if the click was inside the axes
            print(f"{x}, {y} -> {to_colour[rf.predict(np.array([x, y, 1]))]}")

    # Create a figure and an axis
    # Connect the onclick event to the function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    p2 = cProfile.Profile()
    p2.enable()
    model = RandomForestClassifier(n_estimators=200)
    model.fit(x_train, y_train)
    p2.disable()
    stats = pstats.Stats(p2).sort_stats('cumtime')
    stats.print_stats()
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy - sklearn: {accuracy}")

    # p1 = cProfile.Profile()
    # p1.enable()

    result = rf.test(x_test, y_test)

    # p1.disable()
    # stats = pstats.Stats(p1).sort_stats('cumtime')
    # stats.print_stats()

    print(f"Accuracy – {result.accuracy()}")
    print(f"F1-macro - {result.f1.average_score()}")
    print(f"F1-micro - {result.f1.global_score()}")
    print(f"F1-all-labels - {result.f1.labels_score()}")
    print(f"F1-print - {result.f1}")

