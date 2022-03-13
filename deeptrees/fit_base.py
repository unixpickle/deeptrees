from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .tree import Tree


@dataclass
class TreeBuilder(ABC):
    @abstractmethod
    def fit(self, xs: np.ndarray, ys: np.ndarray) -> Tree:
        """
        Fit a tree to the inputs and outputs.

        :param xs: an [N x n_features] array.
        :param ys: an [N x n_outputs] array.
        """
