import itertools
from typing import Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from deeptrees.cascade import Batch, CascadeFlatten, CascadeSGD, CascadeTAO
from deeptrees.cascade_init import (
    CascadeConvInit,
    CascadeGradientLossInit,
    CascadeRawInit,
    CascadeSequentialInit,
    CascadeTAOInit,
)
from deeptrees.experiments.boosting_mnist import dataset_to_tensors
from deeptrees.fit_torch import TorchObliqueBranchBuilder
from deeptrees.gradient_boosting import BoostingSoftmaxLoss
from torchvision.datasets.mnist import MNIST


def main():
    print("loading data...")
    train_dataset = MNIST("./mnist_data", train=True, download=True)
    test_dataset = MNIST("./mnist_data", train=False, download=True)
    xs, ys = dataset_to_tensors(train_dataset, spatial=True)
    test_xs, test_ys = dataset_to_tensors(test_dataset, spatial=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys = xs.to(device), ys.to(device)
    test_xs, test_ys = test_xs.to(device), test_ys.to(device)

    print("initializing TAO model...")
    init_batch_size = 2048
    tao_args = dict(
        tree_depth=2,
        branch_builder=TorchObliqueBranchBuilder(
            max_epochs=1,
            optimizer_kwargs=dict(lr=1e-3, weight_decay=0.01),
        ),
        random_prob=0.1,
        reject_unimprovement=False,
    )
    model, _ = CascadeSequentialInit(
        [
            CascadeConvInit(
                contained=CascadeTAOInit(out_size=16, **tao_args),
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            CascadeConvInit(
                contained=CascadeTAOInit(out_size=32, **tao_args),
                kernel_size=5,
                stride=2,
                padding=1,
            ),
            CascadeRawInit(CascadeFlatten()),
            CascadeTAOInit(out_size=10, **tao_args),
        ]
    ).map(CascadeGradientLossInit)(Batch.with_x(xs[:init_batch_size]))
    sgd_model = CascadeSGD(
        model,
        interval=5,
        opt=optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1),
    )
    print(f"model has {sum(x.numel() for x in model.parameters())} parameters.")

    print("training...")
    loss = BoostingSoftmaxLoss()
    for epoch in itertools.count():
        losses = sgd_model.update(
            full_batch=Batch.with_x(xs),
            loss_fn=lambda indices, batch: loss(batch.x, ys[indices]),
            batch_size=1024,
            outer_batch_size=10000,
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
