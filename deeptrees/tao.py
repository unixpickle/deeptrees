"""
A flexible base implementation of Tree Alternating Optimization:
https://proceedings.neurips.cc/paper/2018/file/185c29dc24325934ee377cfda20e414c-Paper.pdf
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .fit_base import TreeBranchBuilder, TreeBuilder
from .tree import Tree, TreeBranch, TreeLeaf


@dataclass
class TAOResult:
    tree: Tree
    losses: torch.Tensor


@dataclass
class TAOBase(ABC):
    """
    An abstract implementation of Tree Alternating Optimization.

    An instance of this class is intended to be used on a static dataset.
    As such, the instance member `xs` contains the inputs for the dataset, and
    various methods take sample_indices arguments which are Tensors indexing
    into xs.

    This class handles the high-level optimization procedure outlined in:
    https://proceedings.neurips.cc/paper/2018/file/185c29dc24325934ee377cfda20e414c-Paper.pdf

    The actual process of learning decision or leaf nodes and computing losses
    is left to subclasses by overriding build_branch(), build_leaf(), and
    output_loss().
    """

    xs: torch.Tensor

    def optimize(
        self, sub_tree: Tree, sample_indices: Optional[torch.Tensor] = None
    ) -> TAOResult:
        if sample_indices is None:
            sample_indices = torch.arange(len(self.xs))

        if len(sample_indices) == 0:
            return TAOResult(
                tree=sub_tree,
                losses=torch.zeros(0, dtype=self.xs.dtype, device=self.xs.device),
            )

        if isinstance(sub_tree, TreeLeaf):
            return self.build_leaf(sub_tree, sample_indices)

        assert isinstance(sub_tree, TreeBranch)
        xs = self.xs[sample_indices]
        with torch.no_grad():
            decision = sub_tree.decision(xs)
        left_indices = sample_indices[~decision]
        right_indices = sample_indices[decision]

        left_result = self.optimize(sub_tree.left, left_indices)
        right_result = self.optimize(sub_tree.right, right_indices)

        left_losses = torch.zeros(xs.shape[0], dtype=xs.dtype, device=xs.device)
        right_losses = torch.zeros_like(left_losses)
        left_losses[~decision] = left_result.losses
        right_losses[decision] = right_result.losses
        left_losses[decision] = self.loss_at_subtree(left_result.tree, right_indices)
        right_losses[~decision] = self.loss_at_subtree(right_result.tree, left_indices)

        return self.build_branch(
            cur_branch=sub_tree.with_children(left_result.tree, right_result.tree),
            sample_indices=sample_indices,
            left_losses=left_losses,
            right_losses=right_losses,
        )

    def loss_at_subtree(
        self, sub_tree: Tree, sample_indices: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = sub_tree(self.xs[sample_indices])
        return self.output_loss(sample_indices, outputs)

    @abstractmethod
    def build_branch(
        self,
        cur_branch: TreeBranch,
        sample_indices: torch.Tensor,
        left_losses: torch.Tensor,
        right_losses: torch.Tensor,
    ) -> TAOResult:
        """
        Replace a branch node given the samples that reach it and the output
        losses incurred by following each path for each node.
        """

    @abstractmethod
    def build_leaf(self, cur_leaf: TreeLeaf, sample_indices: torch.Tensor) -> TAOResult:
        """
        Replace a leaf node given the samples that reach it, and evaluate the
        resulting loss for all samples that reached the new leaf.
        """

    @abstractmethod
    def output_loss(
        self, sample_indices: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function for a batch of samples and corresponding
        output predictions.

        :param sample_indices: a tensor of shape [N] of indices into self.xs.
        :param outputs: a tensor of shape [N x D] of output vectors per sample.
        :return: a tensor of shape [N] containing losses for each sample.
        """


@dataclass
class StandaloneTAO(TAOBase):
    """
    A concrete subclass of TAOBase that implements a version of TAO with a
    specified loss function, targets, and regression algorithms for branches
    and leaves.
    """

    ys: torch.Tensor
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    leaf_builder: TreeBuilder
    branch_builder: TreeBranchBuilder
    reject_unimprovement: bool = True

    def build_branch(
        self,
        cur_branch: TreeBranch,
        sample_indices: torch.Tensor,
        left_losses: torch.Tensor,
        right_losses: torch.Tensor,
    ) -> TAOResult:
        xs = self.xs[sample_indices]
        tree = self.branch_builder.fit_branch(
            cur_branch=cur_branch,
            xs=xs,
            left_losses=left_losses,
            right_losses=right_losses,
        )
        with torch.no_grad():
            losses = torch.where(tree.decision(xs), right_losses, left_losses)
            if self.reject_unimprovement:
                old_losses = torch.where(
                    cur_branch.decision(xs), right_losses, left_losses
                )
                if old_losses.mean().item() < losses.mean().item():
                    tree = cur_branch
                    losses = old_losses
        return TAOResult(tree=tree, losses=losses)

    def build_leaf(self, cur_leaf: TreeLeaf, sample_indices: torch.Tensor) -> TAOResult:
        _ = cur_leaf
        xs = self.xs[sample_indices]
        ys = self.ys[sample_indices]
        tree = self.leaf_builder.fit(xs, ys)
        with torch.no_grad():
            outputs = tree(xs)
        losses = self.loss_fn(outputs, ys)
        if self.reject_unimprovement:
            with torch.no_grad():
                old_outputs = cur_leaf(xs)
            old_losses = self.loss_fn(old_outputs, ys)
            if old_losses.mean().item() < losses.mean().item():
                tree = cur_leaf
                losses = old_losses
        return TAOResult(tree=tree, losses=losses)

    def output_loss(
        self, sample_indices: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(outputs, self.ys[sample_indices])


@dataclass
class TAOTreeBuilder(TreeBuilder):
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    base_builder: TreeBuilder
    leaf_builder: TreeBuilder
    branch_builder: TreeBranchBuilder
    reject_unimprovement: bool = True
    max_iterations: int = 15
    min_improvement: float = 1e-5
    verbose: bool = False

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> Tree:
        tree = self.base_builder.fit(xs, ys)
        tao = StandaloneTAO(
            xs=xs,
            ys=ys,
            loss_fn=self.loss_fn,
            leaf_builder=self.leaf_builder,
            branch_builder=self.branch_builder,
            reject_unimprovement=self.reject_unimprovement,
        )
        cur_loss = None
        for i in range(self.max_iterations):
            result = tao.optimize(tree)
            tree = result.tree
            loss = result.losses.mean().item()
            if self.verbose:
                print(f"- TAO iteration {i}: loss={loss:.05f}")
            if cur_loss is not None and loss > cur_loss - self.min_improvement:
                break
            cur_loss = loss
        return tree.prune(xs)
