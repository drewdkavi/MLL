import cProfile

from package.Models.ModelsTemplate import Classifier
from package.Models.Classifier.DecisionTrees import generateRule
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
                 rule_splitting_value: float = 0,
                 rule_splitting_dim: int = -1,
                 depth: int = 0):
        self._xs = XS  # this should be a list of (x_train_dp,  y_train_dp) where y is the categories
        self._ys = YS
        self.rule_split = rule_splitting_value  # type: float
        self.rule_dim = rule_splitting_dim  # type: int
        self.NUM_CATEGORIES = NUM_CATEGORIES  # type: int
        self.INPUT_DIM = INPUT_DIM  # type: int
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
            generateRule.generateRule_cy(self._xs, self._ys, self.INPUT_DIM, self.NUM_CATEGORIES
                                         ))
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


class DecisionTree(Classifier):
    """
    Decision Tree Classifier
    ________________________

    Initialisation:
    |-> Pass as a keyword argument the INPUT_DIMENSION, i.e. the number(int) of explanatory variables
    |-> Pass as a kwarg the NUM_OF_CLASSES we classify the data as - an integer also
    """

    def __init__(self, *, input_dimension: int = -1, num_classes: int = 1):
        if input_dimension < 1:
            raise ValueError(f"input_dimension must be specified: {input_dimension}")
        if num_classes < 1:
            raise ValueError(f"Must require classifying to at least 1 class: num_classes = {num_classes}")
        self._INPUT_DIM: int = input_dimension
        self._num_classes: int = num_classes
        self.tree: _Node | None = None

    def train(self, x_train: npt.NDArray[npt.NDArray[float]], y_train: npt.NDArray[int | float]):

        if len(x_train) != len(y_train):
            raise ValueError(
                f"Mismatched dimensions of X_training - dim {len(x_train)}, and Y_training - dim {len(y_train)}")

        root: _Node = _Node(x_train, y_train, INPUT_DIM=self._INPUT_DIM, NUM_CATEGORIES=self._num_classes)

        # TODO: Make this a true stack? e.g. using linked list (poss. package deque?)
        nodeStack: list[_Node] = [root]

        while len(nodeStack) != 0:
            n = nodeStack.pop()
            if not n.isPure():
                xl, yl, xr, yr = n.generateRule()

                left_child = _Node(xl, yl, INPUT_DIM=self._INPUT_DIM, NUM_CATEGORIES=self._num_classes, )
                right_child = _Node(xr, yr, INPUT_DIM=self._INPUT_DIM, NUM_CATEGORIES=self._num_classes, )
                n.leftChild = left_child
                n.rightChild = right_child
                nodeStack.append(right_child)
                nodeStack.append(left_child)

        self.tree = root
        print(f"Tree height: {get_sub_tree_height(self.tree)}")

    def predict(self, x_dp: npt.NDArray[float]) -> int | float | str:
        """Climbs down the binary tree, until we reach a pure node - returning that pure node's classification"""
        current_node: _Node = self.tree
        while not current_node.isPure():
            val, dim = current_node.getRuleTuple()
            if x_dp[dim] <= val:
                current_node = current_node.leftChild
            else:
                current_node = current_node.rightChild

        return next(iter(current_node.categories))
