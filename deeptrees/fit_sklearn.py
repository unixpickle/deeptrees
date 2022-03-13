from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor

from .fit_base import TreeBuilder
from .tree import (
    AxisTreeBranch,
    ConstantTreeLeaf,
    LinearTreeLeaf,
    LowRankTreeLeaf,
    Tree,
)


@dataclass
class SklearnRegressionTreeBuilder(TreeBuilder):
    """
    Build axis-aligned regression trees using sklearn.
    """

    estimator: Optional[DecisionTreeRegressor] = None

    def fit(self, xs: np.ndarray, ys: np.ndarray) -> Tree:
        estimator = (
            clone(self.estimator)
            if self.estimator is not None
            else DecisionTreeRegressor()
        )
        estimator.fit(xs, ys)
        tree = estimator.tree_

        def tree_at_index(i: int) -> Tree:
            left_id, right_id = tree.children_left[i], tree.children_right[i]
            if left_id == right_id:
                return ConstantTreeLeaf(tree.value[i].reshape([-1]))
            else:
                return AxisTreeBranch(
                    left=tree_at_index(left_id),
                    right=tree_at_index(right_id),
                    axis=tree.feature[i],
                    threshold=tree.threshold[i],
                )

        return tree_at_index(0)


@dataclass
class SklearnLinearLeafBuilder(TreeBuilder):
    """
    Learn linear output leaves using an sklearn linear regression model.
    """

    estimator: Optional[Union[LinearRegression, Ridge, RidgeCV]] = None

    def fit(self, xs: np.ndarray, ys: np.ndarray) -> Tree:
        estimator = clone(self.estimator) if self.estimator is not None else RidgeCV()
        estimator.fit(xs, ys)
        return LinearTreeLeaf(coef=estimator.coef_.T, bias=estimator.intercept_)


@dataclass
class SklearnLowRankLeafBuilder(TreeBuilder):
    """
    Learn low-rank linear output leaves using an sklearn linear regression
    model.
    """

    estimator: Optional[Union[LinearRegression, Ridge, RidgeCV]] = None
    rank: int = 1

    def fit(self, xs: np.ndarray, ys: np.ndarray) -> Tree:
        if len(ys.shape) == 1 or ys.shape[1] == 1:
            return SklearnLinearLeafBuilder(estimator=self.estimator).fit(xs, ys)

        pca = PCA(n_components=min(ys.shape[1], self.rank), whiten=True)
        pca.fit(ys)
        transformed_ys = pca.transform(ys)

        estimator = clone(self.estimator) if self.estimator is not None else RidgeCV()
        estimator.fit(xs, transformed_ys)
        return LowRankTreeLeaf(
            coef_contract=estimator.coef_.T,
            bias_contract=estimator.intercept_,
            coef_expand=np.sqrt(pca.explained_variance_[:, None]) * pca.components_,
            bias_expand=pca.mean_,
        )
