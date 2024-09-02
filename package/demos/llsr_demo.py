import random
from package.Models.Regressor.LinearRegression import LeastSquare
import pprint as pp

def demo():
    print("Small demo of linear (least squares) regression")

    # DATA

    NUM_DATA_POINTS = 400
    NUM_FEATURES = 6
    x_train = [[random.uniform(0.0, 1.0) for _ in range(NUM_FEATURES)] for _ in range(NUM_DATA_POINTS)]
    y_train = [sum(x) for x in x_train]

    # Initialise and train the model

    llsr = LeastSquare.LinearLeastSquareRegressionModel(NUM_FEATURES)
    llsr.train(x_train, y_train)

    # Test what the regressor predicts on a single data point

    test_input = [random.uniform(0, 1) for _ in range(NUM_FEATURES)] + [1]
    print(f"On test input: {test_input[:-1]} with sum = {sum(test_input)-1}")
    print(f"Predictions = {llsr.predict(test_input)}")



