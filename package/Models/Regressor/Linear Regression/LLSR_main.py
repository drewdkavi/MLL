import random
from LeastSquare import LinearLeastSquareRegressionModel
import pprint as pp

NUM_DATA_POINTS = 400
NUM_FEATURES = 6

llsr = LinearLeastSquareRegressionModel(NUM_FEATURES)

x_train = [[random.uniform(0.0, 1.0) for _ in range(NUM_FEATURES)] for _ in range(NUM_DATA_POINTS)]
y_train = [sum(x) for x in x_train]

llsr.train(x_train, y_train)

test_input = [random.uniform(0, 1) for _ in range(NUM_FEATURES)] + [1]
pp.pp(test_input)
pp.pp(sum(test_input) - 1)
pp.pp(llsr.predict(test_input))



