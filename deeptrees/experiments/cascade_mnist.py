import itertools
from typing import Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from deeptrees.cascade import Batch, CascadeSGD
from deeptrees.cascade_init import CascadeSequentialInit
from deeptrees.experiments.boosting_mnist import dataset_to_tensors
from deeptrees.fit_torch import TorchObliqueBranchBuilder
from deeptrees.gradient_boosting import BoostingSoftmaxLoss
from torchvision.datasets.mnist import MNIST, FashionMNIST


def main():
    print("loading data...")
    train_dataset = MNIST("./mnist_data", train=True, download=True)
    test_dataset = MNIST("./mnist_data", train=False, download=True)
    xs, ys = dataset_to_tensors(train_dataset)
    test_xs, test_ys = dataset_to_tensors(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys = xs.to(device), ys.to(device)
    test_xs, test_ys = test_xs.to(device), test_ys.to(device)

    print("initializing TAO model...")
    model, _ = CascadeSequentialInit.tao_dense(
        hidden_sizes=(64, 32, 10),
        tree_depth=3,
        branch_builder=TorchObliqueBranchBuilder(max_epochs=50),
        random_prob=0.2,
    )(Batch.with_x(xs))
    sgd_model = CascadeSGD(
        model, interval=5, opt=optim.Adam(model.parameters(), lr=1e-3)
    )

    print("training...")
    loss = BoostingSoftmaxLoss()
    for epoch in itertools.count():
        losses = sgd_model.update(
            full_batch=Batch.with_x(xs),
            loss_fn=lambda indices, batch: loss(batch.x, ys[indices]),
            batch_size=1024,
        )
        with torch.no_grad():
            sgd_model.eval()
            test_out = sgd_model(Batch.with_x(test_xs)).x
            test_losses = loss(test_out, test_ys)
            test_acc = (test_out.argmax(-1) == test_ys).float().mean()
            sgd_model.train()
        print(
            f"epoch {epoch}: train_loss={losses.mean().item():.05} test_loss={test_losses.mean().item():.05} test_acc={test_acc.item():.05}"
        )


if __name__ == "__main__":
    main()
