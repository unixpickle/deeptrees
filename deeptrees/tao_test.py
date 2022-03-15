import warnings

import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeRegressor

from .fit_sklearn import (
    SklearnLinearLeafBuilder,
    SklearnObliqueBranchBuilder,
    SklearnRegressionTreeBuilder,
)
from .tao import StandaloneTAO
from .tree import Tree


def test_standalone_tao():
    xs1 = torch.randn(1000, 3) + torch.tensor([1.0, 1.0, 1.0])
    xs2 = torch.randn(1000, 3) + torch.tensor([-1, -1, -1])
    df1 = xs1 @ torch.tensor([1.0, -1.0, 0.5])
    ys1 = df1 < df1.mean()
    df2 = xs2 @ torch.tensor([1.0, -1.0, 0.5])
    ys2 = df2 < df2.mean()
    xs = torch.cat([xs1, xs2], dim=0)
    ys = torch.cat([ys1, ys2], dim=0)[:, None].float()

    tree = SklearnRegressionTreeBuilder(
        estimator=DecisionTreeRegressor(max_depth=3)
    ).fit(xs, ys)
    tao = StandaloneTAO(
        xs=xs,
        ys=ys,
        loss_fn=lambda x, y: ((x - y) ** 2).view(-1),
        leaf_builder=SklearnLinearLeafBuilder(),
        branch_builder=SklearnObliqueBranchBuilder(),
    )

    def evaluate_loss(tree: Tree) -> float:
        return tao.loss_fn(tree(xs), ys).mean().item()

    prev_loss = evaluate_loss(tree)
    for _ in range(5):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = tao.optimize(tree)
        assert not result.losses.requires_grad
        tree = result.tree
        next_loss = evaluate_loss(tree)
        assert abs(next_loss - result.losses.mean().item()) < 1e-4
        assert next_loss <= prev_loss + 1e-5
        prev_loss = next_loss
