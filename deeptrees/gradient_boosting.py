from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fit_base import TreeBuilder
from .tree import ConstantTreeLeaf, Tree, TreeBranch


class BoostingLoss(nn.Module):
    @abstractmethod
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function for each element of a batch.

        :param outputs: an [N x D] tensor of outputs from the model.
        :param targets: an [N x ...] tensor of target labels.
        :return: an [N] tensor of losses, one per batch element.
        """

    @abstractmethod
    def leaf_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the optimal constant vector to add to outputs to minimize the
        loss.
        """

    @abstractmethod
    def initial_leaf_value(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Get a reasonable default initial leaf value for the targets.

        This method is responsible for inferring the output shape.
        """

    def gradient(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if outputs.requires_grad:
            outputs = outputs.detach()
        else:
            outputs = outputs.clone()
        outputs = outputs.requires_grad_(True)
        return torch.autograd.grad(self(outputs, targets).sum(), (outputs,))[0]


class BoostingMSELoss(BoostingLoss):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ((outputs - targets) ** 2).mean(-1)

    def leaf_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (targets - outputs).mean(0)

    def initial_leaf_value(self, targets: torch.Tensor) -> torch.Tensor:
        return targets.mean(0)


class BoostingSoftmaxLoss(BoostingLoss):
    """
    A softmax loss where targets are encoded categorically or in terms of
    probabilities.
    """

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(outputs, dim=-1)
        if len(targets.shape) == 1:
            return -log_probs[range(len(log_probs)), targets]
        else:
            return (-log_probs * targets).sum(-1)

    def leaf_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = outputs.shape[1]
        residual = -self.gradient(outputs, targets)

        # A single Newton-Raphson step, as in
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/ensemble/_gb_losses.py#L838

        if len(targets.shape) == 1:
            targets = _one_hot(targets, num_classes)

        numerator = residual.sum(0) * (num_classes - 1) / num_classes
        denominator = ((targets - residual) * (1 - (targets - residual))).sum(0)
        return torch.where(
            denominator < 1e-8, torch.zeros_like(numerator), numerator / denominator
        )

    def initial_leaf_value(self, targets: torch.Tensor) -> torch.Tensor:
        if len(targets.shape) == 1:
            targets = _one_hot(targets)
        return (targets.mean(0) + 1e-12).log()


class AdditiveEnsemble(nn.Module):
    def __init__(self, init_output: torch.Tensor, estimators: Sequence[nn.Module] = ()):
        super().__init__()
        self.estimators = nn.ModuleList(estimators)
        self.register_buffer("init_output", init_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.init_output[None].repeat(len(x), 1)
        for module in self.estimators:
            output = output + module(x)
        return output


@dataclass
class GradientBooster:
    builder: TreeBuilder
    loss: BoostingLoss
    learning_rate: float = 1.0
    n_estimators: int = 100
    verbose: bool = False

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> AdditiveEnsemble:
        model = AdditiveEnsemble(init_output=self.loss.initial_leaf_value(ys))
        for i in range(self.n_estimators):
            tree = self.add_tree(model, xs, ys)
            model.estimators.append(tree)
            if self.verbose:
                with torch.no_grad():
                    output = model(xs)
                    loss = self.loss(output, ys).mean().item()
                print(f"iter {i+1}: loss={loss:.05f}")
        return model

    def add_tree(self, model: nn.Module, xs: torch.Tensor, ys: torch.Tensor) -> Tree:
        with torch.no_grad():
            cur_outputs = model(xs)
        residual = -self.loss.gradient(cur_outputs, ys)
        estimator = self.builder.fit(xs, residual)
        return self._build_leaves(estimator, xs, cur_outputs, ys)

    def _build_leaves(
        self, tree: Tree, xs: torch.Tensor, prev_outputs: torch.Tensor, ys: torch.Tensor
    ) -> Tree:
        if isinstance(tree, TreeBranch):
            decisions = tree.decision(xs)
            if (decisions == decisions[0]).all().item():
                # If this is an unused branch, treat it as a leaf.
                return self._build_leaf(prev_outputs, ys)
            return tree.with_children(
                left=self._build_leaves(
                    tree.left, xs[~decisions], prev_outputs[~decisions], ys[~decisions]
                ),
                right=self._build_leaves(
                    tree.right, xs[decisions], prev_outputs[decisions], ys[decisions]
                ),
            )
        else:
            return self._build_leaf(prev_outputs, ys)

    def _build_leaf(
        self, prev_outputs: torch.Tensor, ys: torch.Tensor
    ) -> ConstantTreeLeaf:
        return ConstantTreeLeaf(
            output=self.loss.leaf_value(prev_outputs, ys) * self.learning_rate
        )


def _one_hot(vec: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    if num_classes is None:
        num_classes = vec.max().item() + 1
    return (vec[:, None] == torch.arange(num_classes).to(vec)).float()
