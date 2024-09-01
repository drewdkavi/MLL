import time
from os import system, name

from package.demos import randomForest_demo, snn_breastCancer, snn_2class_blobs, snn_4class_blobs_trial, snn_spiral
from package.demos import decisionTree_demo

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def main():

    decisionTree_demo.demo()
    time.sleep(3)
    clear()

    randomForest_demo.demo()
    time.sleep(3)
    clear()

    snn_breastCancer.demo()
    time.sleep(3)
    clear()

    snn_2class_blobs.demo()
    time.sleep(3)
    clear()

    snn_4class_blobs_trial.demo()
    time.sleep(3)
    clear()

    snn_spiral.demo()


if __name__ == '__main__':
    main()
