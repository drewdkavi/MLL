import time
from os import system, name

from package.demos import (snn_2class_blobs, snn_4class_blobs_trial,
                           snn_breastCancer, snn_spiral, snn_irises, snn_digits)
from package.demos import decisionTree_demo, randomForest_demo, rf_irises


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def main():
    print()
    print("Welcome to ... ")
    print(r" _     ___  _   _      _       _     _  ____  ____  ____  ____ ___  _")
    print(r"/ \__/|\  \//  / \__/|/ \     / \   / \/  _ \/  __\/  _ \/  __\\  \//")
    print(r"| |\/|| \  /   | |\/||| |     | |   | || | //|  \/|| / \||  \/| \  / ")
    print(r"| |  || / /    | |  ||| |_/\  | |_/\| || |_\\|    /| |-|||    / / /  ")
    print(r"\_/  \|/_/     \_/  \|\____/  \____/\_/\____/\_/\_\\_/ \|\_/\_\/_/   ")
    print()
    input("Press enter to see some demonstrations of the library")

    clear()

    decisionTree_demo.demo()
    input("Press enter to continue...")
    clear()

    randomForest_demo.demo()
    input("Press enter to continue...")
    clear()

    snn_breastCancer.demo()
    input("Press enter to continue...")
    clear()

    snn_2class_blobs.demo()
    input("Press enter to continue...")
    clear()

    snn_4class_blobs_trial.demo()
    input("Press enter to continue...")
    clear()

    snn_irises.demo()
    input("Press enter to continue...")
    clear()

    # This is quite accurate despite not being a CNN :)
    # snn_digits.demo()
    # input("Press enter to continue...")
    # clear()

    snn_spiral.demo()


if __name__ == '__main__':
    main()
