from abc import abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn


class Tree(nn.Module):
    @abstractmethod
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict features for inputs.

        :param xs: an [N x n_features] tensor.
        :return: an [N x n_outputs] tensor.
        """


class TreeLeaf(Tree):
    pass


class TreeBranch(Tree):
    def __init__(self, left: Tree, right: Tree):
        super().__init__()
        self.left = left
        self.right = right

    def with_children(self, left: Tree, right: Tree) -> "TreeBranch":
        result = deepcopy(self)
        result.left = left
        result.right = right
        return result

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        decisions = self.decision(xs)
        sub_left = self.left(xs[~decisions])
        sub_right = self.right(xs[decisions])

        out = torch.zeros(len(xs), sub_left.shape[1]).to(sub_left)
        out[~decisions] = sub_left
        out[decisions] = sub_right
        return out

    @abstractmethod
    def decision(self, xs: torch.Tensor) -> torch.Tensor:
        pass


class AxisTreeBranch(TreeBranch):
    def __init__(self, left: Tree, right: Tree, axis: int, threshold: torch.Tensor):
        super().__init__(left, right)
        self.axis = axis
        self.register_buffer("threshold", threshold)

    def decision(self, xs: torch.Tensor) -> torch.Tensor:
        return xs[:, self.axis] > self.threshold


class ObliqueTreeBranch(TreeBranch):
    def __init__(
        self, left: Tree, right: Tree, coef: torch.Tensor, threshold: torch.Tensor
    ):
        super().__init__(left, right)
        self.register_buffer("coef", coef)
        self.register_buffer("threshold", threshold)

    def decision(self, xs: torch.Tensor) -> torch.Tensor:
        return (xs @ self.coef).view(-1) > self.threshold


class LinearTreeLeaf(TreeLeaf):
    def __init__(self, coef: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.coef = nn.Parameter(coef)
        self.bias = nn.Parameter(bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return (xs @ self.coef + self.bias).view(len(xs), self.bias.numel())


class LowRankTreeLeaf(TreeLeaf):
    def __init__(
        self,
        coef_contract: torch.Tensor,
        bias_contract: torch.Tensor,
        coef_expand: torch.Tensor,
        bias_expand: torch.Tensor,
    ):
        super().__init__()
        self.coef_contract = nn.Parameter(coef_contract)
        self.bias_contract = nn.Parameter(bias_contract)
        self.coef_expand = nn.Parameter(coef_expand)
        self.bias_expand = nn.Parameter(bias_expand)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return (
            xs @ self.coef_contract + self.bias_contract
        ) @ self.coef_expand + self.bias_expand


class ConstantTreeLeaf(TreeLeaf):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = nn.Parameter(output)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.output.view(1, -1).repeat(len(xs), 1)
