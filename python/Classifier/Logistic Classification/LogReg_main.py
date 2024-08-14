import random
import numpy as np

from LogisticReg import BinaryLogisticRegression
from python.norm_objects import square_nw
import pprint as pp

NUM_DATA_POINTS = 4000
NUM_FEATURES = 8

blr = BinaryLogisticRegression(NUM_FEATURES)

x_train = [[random.uniform(0, 1) for _ in range(NUM_FEATURES)] for _ in range(NUM_DATA_POINTS)]
y_train = []
for x in x_train:
    if sum(x) < NUM_FEATURES/2:
        y_train.append(1)
    else:
        y_train.append(0)
x_train = [x + [1] for x in x_train]

x_train = np.array(x_train)
y_train = np.array(y_train)

for x, y in zip(x_train, y_train):
    print(sum(x), y)

blr.train(x_train, y_train)

x_test = [[random.uniform(0, 1) for _ in range(NUM_FEATURES)] for _ in range(100)]

y_test = []
for x in x_test:
    if sum(x) < NUM_FEATURES/2:
        y_test.append(1)
    else:
        y_test.append(0)
x_test = [np.array(x + [1]) for x in x_test]
x_test = np.array(x_test)
y_test = np.array(y_test)

print(blr.test(x_test, y_test))




