import math, random
import numpy as np


def get_2blobs_diagonal(N: int, blob_radius=0.23):
    xs = np.zeros((N, 2), dtype=float)
    ys = np.zeros(N, dtype=str)

    for i in range(N):

        y = random.choice([0, 1, 2, 3])

        if y == 0:

            ys[i] = 'Strawberries'
            x1_init = 0.25
            x2_init = 0.75

        elif y == 1:

            ys[i] = 'Apples'
            x1_init = 0.25
            x2_init = 0.25

        elif y == 2:

            ys[i] = 'Strawberries'
            x1_init = 0.75
            x2_init = 0.25

        elif y == 3:

            ys[i] = 'Apples'
            x1_init = 0.75
            x2_init = 0.75
        else:
            x1_init = -1
            x2_init = -1

        r = blob_radius * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 1) * 2 * np.pi
        x1 = x1_init + r * math.cos(theta)
        x2 = x2_init + r * math.sin(theta)

        xs[i][0] = x1
        xs[i][1] = x2

    return xs, ys


def get_4blobs(N: int = 1000, blob_radius: float = 0.25):

        xs = np.zeros((N, 2), dtype=float)
        ys = np.zeros(N, dtype=str)

        for i in range(N):

            y = random.choice([0, 1, 2, 3])
            y = random.choice([0, 1, 2, 3])
            if y == 0:
                ys[i] = 'Strawberries'
                x1_init = 0.3
                x2_init = 0.75

            elif y == 1:
                ys[i] = 'Apples'
                x1_init = 0.25
                x2_init = 0.3

            elif y == 2:
                ys[i] = 'Blueberries'
                x1_init = 0.7
                x2_init = 0.25

            elif y == 3:
                ys[i] = 'Grapes'
                x1_init = 0.75
                x2_init = 0.7
            else:
                x1_init = -1
                x2_init = -1

            r = blob_radius * math.sqrt(random.uniform(0, 1))
            theta = random.uniform(0, 1) * 2 * np.pi
            x1 = x1_init + r * math.cos(theta)
            x2 = x2_init + r * math.sin(theta)

            xs[i][0] = x1
            xs[i][1] = x2
        return xs, ys


def get_twin_spiral_data(N: int, a: float = 0.1, b: float = 0.27): # returns Xs, ys <- where ys is an [int]

        theta = np.linspace(0, 4 * np.pi, N)
        r = a + b * theta
        x11s = r * np.cos(theta)
        x12s = r * np.sin(theta)
        spiral1 = np.vstack((x11s, x12s)).T
        y1 = np.array([0 for _ in range(len(spiral1))])

        x21s = r * np.cos(theta + np.pi)  # Offset the second spiral by 180 degrees
        x22s = r * np.sin(theta + np.pi)
        spiral2 = np.vstack((x21s, x22s)).T
        y2 = np.array([1 for _ in range(len(spiral2))])

        Xs = np.vstack((spiral1, spiral2))
        ys = np.concatenate((y1, y2))

        return Xs, ys

