from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .tree import Tree, TreeBranch


@dataclass
class TreeBuilder(ABC):
    @abstractmethod
    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> Tree:
        """
        Fit a tree to the inputs and outputs.

        :param xs: an [N x n_features] array.
        :param ys: an [N x n_outputs] array.
        """


@dataclass
class TreeBranchBuilder(ABC):
    @abstractmethod
    def fit_branch(
        self,
        cur_branch: TreeBranch,
        xs: torch.Tensor,
        left_losses: torch.Tensor,
        right_losses: torch.Tensor,
    ) -> TreeBranch:
        pass
