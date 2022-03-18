import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor

from .fit_sklearn import SklearnRegressionTreeBuilder
from .gradient_boosting import BoostingSoftmaxLoss, GradientBooster


def test_gradient_boosting():
    dataset = np.concatenate(
        [
            np.random.normal(size=(1000, 2), loc=(2.0, 2.0)),
            np.random.normal(size=(1000, 2), loc=(-1.0, 0.0)),
            np.random.normal(size=(1000, 2), loc=(1.0, 1.0)),
        ]
    )
    labels = np.array([0] * 1000 + [1] * 1000 + [2] * 1000)
    xs = torch.from_numpy(dataset)
    ys = torch.from_numpy(labels)
    booster = GradientBooster(
        builder=SklearnRegressionTreeBuilder(
            estimator=DecisionTreeRegressor(max_depth=3)
        ),
        loss=BoostingSoftmaxLoss(),
        learning_rate=0.5,
        n_estimators=10,
    )
    good_model = booster.fit(xs, ys)
    booster.n_estimators = 1
    bad_model = booster.fit(xs, ys)

    good_loss = booster.loss(good_model(xs), ys).mean().item()
    bad_loss = booster.loss(bad_model(xs), ys).mean().item()

    assert bad_loss > good_loss
