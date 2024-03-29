from abc import ABC, abstractmethod
from typing import Callable, Iterator, Tuple

import torch
import torch.nn as nn


class Tree(nn.Module, ABC):
    @abstractmethod
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict features for inputs.

        :param xs: an [N x n_features] tensor.
        :return: an [N x n_outputs] tensor.
        """
        pass

    @abstractmethod
    def forward_chunks(
        self, indices: torch.Tensor, xs: torch.Tensor
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the model to the inputs and yield chunks of outputs.

        This must yield at least one chunk, and all indices must be used
        exactly once.

        :param indices: a tensor of shape [N] with indices for each sample.
        :param xs: an [N x n_features] tensor.
        :return: an iterator over (sub_indices, sub_outputs). The sub_indices
                 should be non-overlapping subsets of indices.
        """

    @abstractmethod
    def prune(self, xs: torch.Tensor) -> "Tree":
        """
        Create a version of self such that all branches are taken by at least
        one sample in xs.
        """

    @abstractmethod
    def map_branches(self, fn: Callable[["TreeBranch"], "TreeBranch"]) -> "Tree":
        pass

    @abstractmethod
    def map_leaves(self, fn: Callable[["TreeLeaf"], "TreeLeaf"]) -> "Tree":
        pass

    @abstractmethod
    def iterate_leaves(self) -> Iterator["TreeLeaf"]:
        pass


class TreeLeaf(Tree):
    def forward_chunks(
        self, indices: torch.Tensor, xs: torch.Tensor
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        yield indices, self(xs)

    def prune(self, xs: torch.Tensor) -> Tree:
        _ = xs
        return self

    def map_branches(self, fn: Callable[["TreeBranch"], "TreeBranch"]) -> "Tree":
        _ = fn
        return self

    def map_leaves(self, fn: Callable[["TreeLeaf"], "TreeLeaf"]) -> "Tree":
        return fn(self)

    def iterate_leaves(self) -> Iterator["TreeLeaf"]:
        yield self


class TreeBranch(Tree):
    def __init__(self, left: Tree, right: Tree):
        super().__init__()
        self.left = left
        self.right = right

    @abstractmethod
    def with_children(self, left: Tree, right: Tree) -> "TreeBranch":
        """
        Create a shallow copy of self with different children.
        """

    @abstractmethod
    def decision(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean tensor of shape [N] for an input batch [N x ...].
        The True values go to the right branch, and False to the left.
        """

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        result = None
        for indices, outs in self.forward_chunks(
            torch.arange(len(xs), device=xs.device), xs
        ):
            if result is None:
                result = torch.empty(
                    (len(xs), *outs.shape[1:]), device=outs.device, dtype=outs.dtype
                )
            result[indices] = outs
        return result

    def forward_chunks(
        self, indices: torch.Tensor, xs: torch.Tensor
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        decisions = self.decision(xs)
        yield from self.left.forward_chunks(indices[~decisions], xs[~decisions])
        yield from self.right.forward_chunks(indices[decisions], xs[decisions])

    def prune(self, xs: torch.Tensor) -> Tree:
        decisions = self.decision(xs)
        if (decisions == 0).all().item():
            return self.left
        elif (decisions == 1).all().item():
            return self.right
        return self

    def map_branches(self, fn: Callable[["TreeBranch"], "TreeBranch"]) -> "Tree":
        return fn(
            self.with_children(self.left.map_branches(fn), self.right.map_branches(fn))
        )

    def map_leaves(self, fn: Callable[["TreeLeaf"], "TreeLeaf"]) -> "Tree":
        return self.with_children(self.left.map_leaves(fn), self.right.map_leaves(fn))

    def iterate_leaves(self) -> Iterator["TreeLeaf"]:
        yield from self.left.iterate_leaves()
        yield from self.right.iterate_leaves()


class ObliqueTreeBranch(TreeBranch):
    def __init__(
        self,
        left: Tree,
        right: Tree,
        coef: torch.Tensor,
        threshold: torch.Tensor,
        random_prob: float = 0.0,
    ):
        super().__init__(left, right)
        self.random_prob = random_prob
        self.register_buffer("coef", coef)
        self.register_buffer("threshold", threshold)

    def with_children(self, left: Tree, right: Tree) -> "TreeBranch":
        return ObliqueTreeBranch(
            left=left,
            right=right,
            coef=self.coef,
            threshold=self.threshold,
            random_prob=self.random_prob,
        )

    def decision(self, xs: torch.Tensor) -> torch.Tensor:
        raw_output = (xs @ self.coef).view(-1) > self.threshold
        if self.training and self.random_prob > 0:
            mask = torch.rand(len(xs), device=xs.device) > self.random_prob
            randomized = torch.randint(
                low=0, high=2, size=(len(xs),), device=xs.device
            ).bool()
            return torch.where(mask, raw_output, randomized)
        else:
            return raw_output


class AxisTreeBranch(TreeBranch):
    def __init__(self, left: Tree, right: Tree, axis: int, threshold: torch.Tensor):
        super().__init__(left, right)
        self.axis = axis
        self.register_buffer("threshold", threshold)

    def with_children(self, left: Tree, right: Tree) -> "TreeBranch":
        return AxisTreeBranch(
            left=left, right=right, axis=self.axis, threshold=self.threshold
        )

    def decision(self, xs: torch.Tensor) -> torch.Tensor:
        return xs[:, self.axis] > self.threshold

    def to_oblique(self, xs: torch.Tensor, random_prob: float) -> ObliqueTreeBranch:
        coef = torch.zeros(xs.shape[1], dtype=xs.dtype, device=xs.device)
        coef[self.axis] = 1.0
        return ObliqueTreeBranch(
            left=self.left,
            right=self.right,
            coef=coef,
            threshold=self.threshold,
            random_prob=random_prob,
        )


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


class GateTreeLeaf(TreeLeaf):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = nn.Parameter(weights)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return xs * (self.weights.tanh() + 1)
