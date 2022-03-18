import argparse
import os
import pickle
from typing import Tuple, Union

import numpy as np
import torch
from deeptrees.fit_base import ConstantLeafBuilder
from deeptrees.fit_sklearn import (
    SklearnObliqueBranchBuilder,
    SklearnRegressionTreeBuilder,
)
from deeptrees.gradient_boosting import BoostingSoftmaxLoss, GradientBooster
from deeptrees.tao import TAOTreeBuilder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor
from torchvision.datasets.mnist import MNIST, FashionMNIST

OUTPUT_DIR = "./models_boosting_mnist"


def main():
    print("loading data...")
    train_dataset = MNIST("./mnist_data", train=True, download=True)
    test_dataset = MNIST("./mnist_data", train=False, download=True)
    xs, ys = dataset_to_tensors(train_dataset)
    test_xs, test_ys = dataset_to_tensors(test_dataset)

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
    booster = GradientBooster(
        builder=TAOTreeBuilder(
            loss_fn=lambda x, y: ((x - y) ** 2).mean(-1),
            base_builder=SklearnRegressionTreeBuilder(
                estimator=DecisionTreeRegressor(max_depth=4)
            ),
            leaf_builder=ConstantLeafBuilder(),  # boosting uses constant leaves
            branch_builder=SklearnObliqueBranchBuilder(estimator=LinearSVC(dual=False)),
            min_improvement=0.0003,
            verbose=True,
        ),
        loss=BoostingSoftmaxLoss(),
        learning_rate=0.5,
        n_estimators=20,
        verbose=True,
    )
    tao_model = booster.fit(xs, ys)
    preds = tao_model(test_xs).argmax(-1)
    acc = (preds == test_ys).float().mean()
    print(f" => TAO ensemble accuracy: {acc}")
    with open(os.path.join(OUTPUT_DIR, "tao_ensemble.pkl"), "wb") as f:
        pickle.dump(tao_model, f)


def dataset_to_tensors(
    dataset: Union[MNIST, FashionMNIST]
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    for i in range(len(dataset)):
        img, target = dataset[i]
        images.append(
            torch.from_numpy(np.array(img.convert("RGB"))[..., 0]).view(-1).float()
            / 255
        )
        labels.append(target)
    return torch.stack(images, dim=0), torch.tensor(labels)


if __name__ == "__main__":
    main()
