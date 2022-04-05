import os
import pickle

import numpy as np
import torch
from deeptrees.experiments.data import load_mnist
from deeptrees.fit_base import ConstantLeafBuilder
from deeptrees.fit_sklearn import SklearnRegressionTreeBuilder
from deeptrees.fit_torch import TorchObliqueBranchBuilder
from deeptrees.gradient_boosting import BoostingSoftmaxLoss, GradientBooster
from deeptrees.tao import TAOTreeBuilder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

OUTPUT_DIR = "./models_boosting_mnist"


def main():
    print("loading data...")
    xs, ys = load_mnist(train=True)
    test_xs, test_ys = load_mnist(train=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("fitting sklearn ensemble...")
    sklearn_model = GradientBoostingClassifier(
        max_depth=4,
        n_estimators=20,
        learning_rate=0.5,
        verbose=True,
    )
    sklearn_model.fit(xs.numpy(), ys.numpy())
    preds = sklearn_model.predict(test_xs.numpy())
    acc = np.mean((preds == test_ys.numpy()).astype(np.float32))
    print(f" => sklearn accuracy {acc}")
    with open(os.path.join(OUTPUT_DIR, "sklearn_ensemble.pkl"), "wb") as f:
        pickle.dump(sklearn_model, f)

    print("fitting TAO ensemble...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    booster = GradientBooster(
        builder=TAOTreeBuilder(
            loss_fn=lambda x, y: ((x - y) ** 2).mean(-1),
            base_builder=SklearnRegressionTreeBuilder(
                estimator=DecisionTreeRegressor(max_depth=4)
            ),
            leaf_builder=ConstantLeafBuilder(),  # boosting uses constant leaves
            branch_builder=TorchObliqueBranchBuilder(),
            min_improvement=0.0003,
            verbose=True,
        ),
        loss=BoostingSoftmaxLoss(),
        learning_rate=0.5,
        n_estimators=20,
        verbose=True,
    )
    tao_model = booster.fit(xs.to(device), ys.to(device))
    preds = tao_model(test_xs.to(device)).argmax(-1).cpu()
    acc = (preds == test_ys).float().mean()
    print(f" => TAO ensemble accuracy: {acc}")
    with open(os.path.join(OUTPUT_DIR, "tao_ensemble.pkl"), "wb") as f:
        pickle.dump(tao_model, f)


if __name__ == "__main__":
    main()
