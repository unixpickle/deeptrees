from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Tree(ABC):
    @abstractmethod
    def predict(self, xs: np.ndarray) -> np.ndarray:
        """
        Predict features for inputs.

        :param xs: an [N x n_features] array.
        :return: an [N x n_outputs] array.
        """


@dataclass
class TreeBranch(Tree):
    left: Tree
    right: Tree

    def predict(self, xs: np.ndarray) -> np.ndarray:
        decisions = self.decision(xs)
        sub_left = self.left.predict(xs[np.logical_not(decisions)])
        sub_right = self.right.predict(xs[decisions])
        out = np.zeros_like(sub_left, shape=(len(xs), sub_left.shape[1]))
        out[np.logical_not(decisions)] = sub_left
        out[decisions] = sub_right
        return out

    @abstractmethod
    def decision(self, xs: np.ndarray) -> np.ndarray:
        pass


@dataclass
class AxisTreeBranch(TreeBranch):
    axis: int
    threshold: np.ndarray

    def decision(self, xs: np.ndarray) -> np.ndarray:
        return xs[:, self.axis] > self.threshold


@dataclass
class ObliqueTreeBranch(TreeBranch):
    coef: np.ndarray
    threshold: np.ndarray

    def decision(self, xs: np.ndarray) -> np.ndarray:
        return xs @ self.coef > self.threshold


@dataclass
class LinearTreeLeaf(Tree):
    coef: np.ndarray
    bias: np.ndarray

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return (xs @ self.coef + self.bias).reshape([len(xs), -1])


@dataclass
class LowRankTreeLeaf(Tree):
    coef_contract: np.ndarray
    bias_contract: np.ndarray
    coef_expand: np.ndarray
    bias_expand: np.ndarray

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return (
            xs @ self.coef_contract + self.bias_contract
        ) @ self.coef_expand + self.bias_expand


@dataclass
class ConstantTreeLeaf(Tree):
    output: np.ndarray

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return np.tile(self.output.reshape([1, -1]), [len(xs), 1])
