from package.Models.Classifier.Random_Forest import generateRuleRF
from package.Models.ModelsTemplate import Classifier
import numpy as np
import numpy.typing as npt
import pstats


def f8_alt(x):
    return "%14.9f" % x


pstats.f8 = f8_alt

'''
Overview:
Natively supports multiclass classification
Inherently classification
Essentially keeps drawing hyperplanes over and over again until hyperplanes can kind of classify into separate classes
|-> this is a explanation

Greedily picks the 'splitting rule' that minimises Gini Impurity (can use other minimisation quanitities)
'''
# TODO: Why do we only consider hyperplane rules? - Can we consider other rules,
#  i.e. non linear-kernels, like one can do in the case of SVMs
#  research academic papers about this!

'''

Structure:

class Node that contains a set (x_train_datapoint, y_train, datapoint) tuples - zipped together:
Each Node except the leaves contain a rule: a binary rule, which splits a nodes dataset into its children
We recursively split nodes until a node is pure, we split by <= or => rules for a dimension in x_train
|-> a node is pure if all of its y_train is of 1 class.
As convention we have the 'if' condition of the rule being our left child, and 'else' being our right
This process clearly builds a tree which classifies in the obvious way.

How do we chose which node to use at each splitting level? We iterate over each value in our training dataset for each
each input dimension in x_train.
Picking the rule at each node which minimizes the weighted gini-impurity.

'''



class _Node:

    def __init__(self, XS, YS,
                 INPUT_DIM: int = -1,
                 NUM_CATEGORIES: int = 0,
                 COLSAMPLE_SIZE: int = -1,
                 rule_splitting_value: float = 0,
                 rule_splitting_dim: int = -1,
                 depth: int = 0):
        self._xs = XS  # this should be a list of (x_train_dp,  y_train_dp) where y is the categories
        self._ys = YS
        self.rule_split = rule_splitting_value  # type: float
        self.rule_dim = rule_splitting_dim  # type: int
        self.NUM_CATEGORIES = NUM_CATEGORIES  # type: int
        self.INPUT_DIM = INPUT_DIM  # type: int
        self.COLSAMPLING_SIZE = COLSAMPLE_SIZE
        self.categories = self.__getCategories()
        self.leftChild: None | _Node = None
        self.rightChild: None | _Node = None
        self.depth: int = depth

    def __getCategories(self):
        return np.unique(self._ys)

    def getRuleTuple(self) -> (float, int):
        return self.rule_split, self.rule_dim

    def generateRule(self):
        # profile = cProfile.Profile()
        # profile.enable()

        xs_l, ys_l, xs_r, ys_r, split_val, split_dim = (

            generateRuleRF.generateRuleRF_cy(
                self._xs, self._ys, self.INPUT_DIM, self.NUM_CATEGORIES, self.COLSAMPLING_SIZE
            )
        )

        self.rule_split = split_val
        self.rule_dim = split_dim

        # profile.disable()
        # stats = pstats.Stats(profile).sort_stats('cumtime')
        # stats.print_stats()

        return xs_l, ys_l, xs_r, ys_r

    def isPure(self) -> bool:
        return len(self.categories) == 1


def get_sub_tree_height(n: _Node | None) -> int:
    if n is None:
        return 0
    else:
        return max(get_sub_tree_height(n.leftChild) + 1, get_sub_tree_height(n.rightChild) + 1)


def boostrapChoice(size: int) -> npt.NDArray[int]:
    return np.random.choice(np.arange(size), size=size)


class RotationForest(Classifier):
    """
    Decision Tree Classifier
    ________________________

    Initialisation:
    |-> Pass as a keyword argument the INPUT_DIMENSION, i.e. the number(int) of explanatory variables
    |-> Pass as a kwarg the NUM_OF_CLASSES we classify the data as - an integer also
    """

    def __init__(
            self,
            input_dimension: int,
            num_classes: int,
            *,
            NUM_ESTIMATORS: int = 20,
            COL_SUBSAMPLE_SIZE: int | None = None, ):

        if input_dimension < 1:
            raise ValueError(f"input_dimension must be specified: {input_dimension}")
        if num_classes < 1:
            raise ValueError(f"Must require classifying to at least 1 class: num_classes = {num_classes}")
        self._INPUT_DIM: int = input_dimension
        self._num_classes: int = num_classes
        self.num_estimators: int = NUM_ESTIMATORS
        self.COL_SUBSAMPLE_SIZE: int = (input_dimension // 10 + 1) if COL_SUBSAMPLE_SIZE is None else COL_SUBSAMPLE_SIZE
        self.trees: npt.NDArray[_Node | None] = np.empty(NUM_ESTIMATORS, dtype=object)

    def _make_tree(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):

        root: _Node = _Node(
            x_train,
            y_train,
            INPUT_DIM=self._INPUT_DIM,
            NUM_CATEGORIES=self._num_classes,
            COLSAMPLE_SIZE=self.COL_SUBSAMPLE_SIZE
        )

        # TODO: Make this a true stack? e.g. using linked list (poss. package deque?)
        nodeStack: list[_Node] = [root]

        while len(nodeStack) != 0:
            n = nodeStack.pop()
            if not n.isPure():
                xl, yl, xr, yr = n.generateRule()

                left_child = _Node(
                    xl, yl,
                    INPUT_DIM=self._INPUT_DIM, NUM_CATEGORIES=self._num_classes,
                    COLSAMPLE_SIZE=self.COL_SUBSAMPLE_SIZE)

                right_child = _Node(
                    xr, yr,
                    INPUT_DIM=self._INPUT_DIM, NUM_CATEGORIES=self._num_classes,
                    COLSAMPLE_SIZE=self.COL_SUBSAMPLE_SIZE)

                n.leftChild = left_child
                n.rightChild = right_child
                nodeStack.append(right_child)
                nodeStack.append(left_child)
        return root

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):

        N: int = len(x_train)

        # TODO: Figure out parallelism
        # def _parallel_train(n):
        #     bootstrapped_indicies = boostrapChoice(N)
        #     x_train_bootstrapped = x_train[bootstrapped_indicies]
        #     y_train_bootstrapped = y_train[bootstrapped_indicies]
        #     return self._make_tree(x_train_bootstrapped, y_train_bootstrapped)
        #
        # pool = Pool()
        # self.trees = np.array(pool.map(_parallel_train, range(self.num_estimators)))

        for i in range(self.num_estimators):
            print(f"Training tree - {i}/{self.num_estimators}", end='\r')
            bootstrapped_indicies = boostrapChoice(N)
            x_train_bootstrapped, y_train_bootstrapped = x_train[bootstrapped_indicies], y_train[bootstrapped_indicies]
            tree_i = self._make_tree(x_train_bootstrapped, y_train_bootstrapped)
            self.trees[i] = tree_i

    def _predict_tree(self, x_dp: npt.NDArray[float], estimator_index: int) -> int | float | str:
        """Climbs down the binary tree, until we reach a pure node - returning that pure node's classification"""
        current_node: _Node = self.trees[estimator_index]
        while not current_node.isPure():
            val, dim = current_node.getRuleTuple()
            if x_dp[dim] <= val:
                current_node = current_node.leftChild
            else:
                current_node = current_node.rightChild

        return next(iter(current_node.categories))

    def predict(self, x_dp: npt.NDArray[float]) -> int | float | str:

        preds = np.zeros(self._num_classes)

        for i in range(self.num_estimators):
            estimator_prediction = self._predict_tree(x_dp, i)
            preds[estimator_prediction] += 1

        # profile.disable()
        # stats = pstats.Stats(profile).sort_stats('cumtime')
        # stats.print_stats()

        return np.argmax(preds)

