import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from .fit_sklearn import SklearnLowRankLeafBuilder, SklearnRegressionTreeBuilder


def test_low_rank_leaf_builder_redundant():
    xs = np.random.normal(size=(1000, 3))
    outputs_base = xs @ np.array([1, 2, 3])
    ys = np.stack([outputs_base, outputs_base * -2, outputs_base * -0.5], axis=1)

    fit_leaf = SklearnLowRankLeafBuilder(estimator=LinearRegression()).fit(xs, ys)
    pred = fit_leaf.predict(xs)

    assert np.mean((pred - ys) ** 2) < 1e-5


def test_low_rank_leaf_builder_non_redundant():
    xs = np.random.normal(size=(1000, 3))

    # Output is actually rank 2
    ys = xs @ np.array([[1, 2, 3], [3, 2, 1], [1, 1, 1]]).T

    # Rank 1 should not work well
    fit_leaf = SklearnLowRankLeafBuilder(estimator=LinearRegression()).fit(xs, ys)
    pred = fit_leaf.predict(xs)
    assert np.mean((pred - ys) ** 2) > 1e-5

    # Rank 2 should be a perfect approximation.
    fit_leaf = SklearnLowRankLeafBuilder(estimator=LinearRegression(), rank=2).fit(
        xs, ys
    )
    pred = fit_leaf.predict(xs)
    assert np.mean((pred - ys) ** 2) < 1e-5


def test_regression_tree_builder():
    xs = np.random.normal(size=(1000, 3))
    ys = xs @ np.array([[1, 2, 3], [3, 2, 1], [1, 1, 1]]).T

    # Make sure deep tree is perfect.
    fit_tree = SklearnRegressionTreeBuilder().fit(xs, ys)
    pred = fit_tree.predict(xs)
    assert np.allclose(pred, ys)

    # Make sure shallow tree is not perfect.
    fit_tree = SklearnRegressionTreeBuilder(
        estimator=DecisionTreeRegressor(max_depth=3)
    ).fit(xs, ys)
    pred = fit_tree.predict(xs)
    assert not np.allclose(pred, ys)
