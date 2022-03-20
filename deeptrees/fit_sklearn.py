from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor

from .fit_base import TreeBranchBuilder, TreeBuilder
from .tree import (
    AxisTreeBranch,
    ConstantTreeLeaf,
    LinearTreeLeaf,
    LowRankTreeLeaf,
    ObliqueTreeBranch,
    Tree,
    TreeBranch,
)


@dataclass
class SklearnRegressionTreeBuilder(TreeBuilder):
    """
    Build axis-aligned regression trees using sklearn.
    """

    estimator: Optional[DecisionTreeRegressor] = None

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> Tree:
        estimator = (
            clone(self.estimator)
            if self.estimator is not None
            else DecisionTreeRegressor()
        )
        estimator.fit(xs.detach().cpu().numpy(), ys.detach().cpu().numpy())
        tree = estimator.tree_

        def tree_at_index(i: int) -> Tree:
            left_id, right_id = tree.children_left[i], tree.children_right[i]
            if left_id == right_id:
                return ConstantTreeLeaf(torch.from_numpy(tree.value[i]).view(-1).to(xs))
            else:
                return AxisTreeBranch(
                    left=tree_at_index(left_id),
                    right=tree_at_index(right_id),
                    axis=tree.feature[i],
                    threshold=torch.tensor(float(tree.threshold[i])).to(xs),
                )

        return tree_at_index(0)


@dataclass
class SklearnObliqueBranchBuilder(TreeBranchBuilder):
    """
    Learn oblique (i.e. linear) splits for a decision tree.
    """

    estimator: Optional[Union[LinearSVC, LogisticRegression]] = None

    def fit_branch(
        self,
        cur_branch: TreeBranch,
        xs: torch.Tensor,
        left_losses: torch.Tensor,
        right_losses: torch.Tensor,
    ) -> TreeBranch:
        estimator = clone(self.estimator) if self.estimator is not None else LinearSVC()

        classes = (right_losses < left_losses).cpu().numpy()
        if (classes == 0).all() or (classes == 1).all():
            return cur_branch

        sample_weight = (left_losses - right_losses).abs().detach().cpu().numpy()
        mean = np.mean(sample_weight)
        if mean == 0:
            return cur_branch
        sample_weight /= mean

        if (
            np.mean(sample_weight[classes]) < 1e-5
            or np.mean(sample_weight[~classes]) < 1e-5
        ):
            # The SVM solver doesn't when one class has near-zero weight.
            return cur_branch

        estimator.fit(
            xs.detach().cpu().numpy(), classes.astype(bool), sample_weight=sample_weight
        )
        return ObliqueTreeBranch(
            left=cur_branch.left,
            right=cur_branch.right,
            coef=torch.from_numpy(estimator.coef_.T).to(xs),
            threshold=torch.tensor(
                -np.array(estimator.intercept_).reshape([]).tolist()
            ).to(xs),
        )


@dataclass
class SklearnLinearLeafBuilder(TreeBuilder):
    """
    Learn linear output leaves using an sklearn linear regression model.
    """

    estimator: Optional[Union[LinearRegression, Ridge, RidgeCV]] = None

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> Tree:
        estimator = clone(self.estimator) if self.estimator is not None else RidgeCV()
        if len(ys.shape) == 2 and ys.shape[1] == 1:
            ys = ys.reshape([-1])
        estimator.fit(xs.detach().cpu().numpy(), ys.detach().cpu().numpy())
        return LinearTreeLeaf(
            coef=torch.from_numpy(estimator.coef_.T).to(xs),
            bias=torch.from_numpy(np.array(estimator.intercept_)).to(xs),
        )


@dataclass
class SklearnLowRankLeafBuilder(TreeBuilder):
    """
    Learn low-rank linear output leaves using an sklearn linear regression
    model.
    """

    estimator: Optional[Union[LinearRegression, Ridge, RidgeCV]] = None
    rank: int = 1

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> Tree:
        if len(ys.shape) == 1 or ys.shape[1] == 1:
            return SklearnLinearLeafBuilder(estimator=self.estimator).fit(xs, ys)

        pca = PCA(n_components=min(ys.shape[1], self.rank), whiten=True)
        pca.fit(ys.detach().cpu().numpy())
        transformed_ys = pca.transform(ys.detach().cpu().numpy())

        estimator = clone(self.estimator) if self.estimator is not None else RidgeCV()
        estimator.fit(xs.detach().cpu().numpy(), transformed_ys)
        return LowRankTreeLeaf(
            coef_contract=torch.from_numpy(estimator.coef_.T).to(xs),
            bias_contract=torch.from_numpy(estimator.intercept_).to(xs),
            coef_expand=torch.from_numpy(
                np.sqrt(pca.explained_variance_[:, None]) * pca.components_
            ).to(xs),
            bias_expand=torch.from_numpy(pca.mean_).to(xs),
        )
