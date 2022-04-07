from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .fit_base import TreeBranchBuilder
from .tree import ObliqueTreeBranch, Tree, TreeBranch


@dataclass
class TorchObliqueBranchBuilder(TreeBranchBuilder):
    """
    Learn oblique splits for a decision tree in a greedy binary classification
    fashion. The model is trained with the SVM hinge loss by default, and will
    save optimization state in the generate branches to get a warm start.
    """

    optimizer: Callable[..., optim.Optimizer] = optim.AdamW
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(lr=1e-3, weight_decay=0.01)
    )
    converge_epochs: int = 10
    max_epochs: int = 1000
    max_iters: Optional[int] = None
    batch_size: Optional[int] = 1024
    warm_start: bool = True
    reset_optimizer: bool = False
    cross_entropy_loss: bool = False

    def fit_branch(
        self,
        cur_branch: TreeBranch,
        xs: torch.Tensor,
        left_losses: torch.Tensor,
        right_losses: torch.Tensor,
    ) -> TreeBranch:
        xs = xs.detach()
        left_losses = left_losses.detach()
        right_losses = right_losses.detach()

        classes = right_losses < left_losses
        if (classes == 0).all() or (classes == 1).all():
            return cur_branch

        sample_weight = (left_losses - right_losses).abs()
        mean = sample_weight.mean().item()
        if mean == 0:
            return cur_branch
        sample_weight /= mean

        if (
            sample_weight[classes].mean() < 1e-5
            or sample_weight[~classes].mean() < 1e-5
        ):
            # The SVM solver doesn't when one class has near-zero weight.
            return cur_branch

        if self.warm_start and isinstance(cur_branch, _StatefulObliqueTreeBranch):
            # These parameters are tied to the optimizer.
            weight = cur_branch.state["weight"]
            bias = cur_branch.state["bias"]
            opt = cur_branch.state["opt"]
        else:
            weight = nn.Parameter(torch.zeros_like(xs[0]))
            bias = nn.Parameter(torch.zeros_like(weight[0]))
            if self.warm_start and isinstance(cur_branch, ObliqueTreeBranch):
                with torch.no_grad():
                    weight.copy_(cur_branch.coef)
                    bias.copy_(cur_branch.threshold)
            opt = None

        if opt is None or self.reset_optimizer:
            opt = self.optimizer([weight, bias], **self.optimizer_kwargs)

        if self.batch_size is None:
            batch_size = len(xs)
        else:
            batch_size = min(self.batch_size, len(xs))

        def accuracy():
            preds = xs @ weight - bias
            loss = torch.relu(1 - preds * (batch_ys.float() * 2 - 1)).view(-1)
            accuracies = ((preds > 0) == classes).float()
            return (
                accuracies.mean().item(),
                ((accuracies * sample_weight).sum() / sample_weight.sum()).item(),
                loss.mean().item(),
                ((loss * sample_weight).sum() / sample_weight.sum()).item(),
            )

        print("pre acc", accuracy())

        history = []
        iters = 0
        for _ in range(self.max_epochs):
            if batch_size < len(xs):
                indices = torch.randperm(len(xs))
                epoch_xs = xs[indices]
                epoch_ys = classes[indices].float()
                epoch_weight = sample_weight[indices]
            else:
                epoch_xs = xs
                epoch_ys = classes
                epoch_weight = sample_weight
            total_loss = 0
            for i in range(0, len(epoch_xs), batch_size):
                batch_xs = epoch_xs[i : i + batch_size]
                batch_ys = epoch_ys[i : i + batch_size]
                batch_weight = epoch_weight[i : i + batch_size]
                preds = batch_xs @ weight - bias

                if self.cross_entropy_loss:
                    loss = F.binary_cross_entropy_with_logits(
                        preds.view(-1), batch_ys.view(-1).float(), reduction="none"
                    )
                else:
                    # Hinge loss
                    loss = torch.relu(1 - preds * (batch_ys.float() * 2 - 1)).view(-1)

                loss_sum = (loss * batch_weight).sum()
                loss_mean = loss_sum / batch_weight.sum()
                opt.zero_grad()
                loss_mean.backward()
                opt.step()
                total_loss += loss_sum.item()
                iters += 1
                if self.max_iters and iters >= self.max_iters:
                    break
            history.append(total_loss)
            if self.should_terminate(history) or (
                self.max_iters and iters >= self.max_iters
            ):
                break

        print("post acc", accuracy())

        kwargs = dict(
            left=cur_branch.left,
            right=cur_branch.right,
            coef=weight.detach().clone(),
            threshold=bias.detach().clone(),
        )
        if self.warm_start:
            return _StatefulObliqueTreeBranch(
                state=dict(
                    weight=weight,
                    bias=bias,
                    opt=opt,
                ),
                **kwargs,
            )
        else:
            return ObliqueTreeBranch(**kwargs)

    def should_terminate(self, history: List[float]) -> bool:
        if len(history) <= self.converge_epochs:
            return False
        prev_mean = (
            sum(history[-(self.converge_epochs + 1) : -1]) / self.converge_epochs
        )
        return history[-1] > prev_mean


class _StatefulObliqueTreeBranch(ObliqueTreeBranch):
    def __init__(
        self, state: Dict[str, Union[optim.Optimizer, nn.Parameter]], **kwargs
    ):
        super().__init__(**kwargs)
        self.state = state

    def with_children(self, left: Tree, right: Tree) -> "TreeBranch":
        return _StatefulObliqueTreeBranch(
            state=self.state,
            left=left,
            right=right,
            coef=self.coef,
            threshold=self.threshold,
            random_prob=self.random_prob,
        )
