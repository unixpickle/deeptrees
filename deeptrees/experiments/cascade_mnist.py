import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from deeptrees.cascade import Batch, CascadeFlatten, CascadeFn, CascadeSGD
from deeptrees.cascade_init import (
    CascadeConvInit,
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
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            CascadeConvInit(
                contained=CascadeTAOInit(out_size=32, **tao_args),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            CascadeRawInit(CascadeFn(nn.MaxPool2d(2))),
            CascadeRawInit(CascadeFlatten()),
            CascadeTAOInit(out_size=128, **tao_args),
            CascadeTAOInit(out_size=10, **tao_args),
        ]
    )(Batch.with_x(xs[:init_batch_size]))
    sgd_model = CascadeSGD(
        model,
        interval=5,
        opt=optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1),
    )
    print(f"model has {sum(x.numel() for x in model.parameters())} parameters.")

    print("training...")
    loss = BoostingSoftmaxLoss()
    for epoch in itertools.count():
        batch_size = 1024
        losses = sgd_model.update(
            full_batch=Batch.with_x(xs),
            loss_fn=lambda indices, batch: loss(batch.x, ys[indices]),
            batch_size=batch_size,
            outer_batch_size=10000,
        )
        with torch.no_grad():
            sgd_model.eval()
            total_test_loss = 0.0
            total_test_correct = 0.0
            for i in range(0, len(test_xs), batch_size):
                test_out = sgd_model(Batch.with_x(test_xs[i : i + batch_size])).x
                test_losses = loss(test_out, test_ys[i : i + batch_size])
                total_test_correct += (
                    (test_out.argmax(-1) == test_ys[i : i + batch_size])
                    .long()
                    .sum()
                    .item()
                )
                total_test_loss += test_losses.sum().item()
            test_loss = total_test_loss / len(test_xs)
            test_acc = total_test_correct / len(test_xs)
            sgd_model.train()
        print(
            f"epoch {epoch}: train_loss={losses.mean().item():.05} test_loss={test_loss:.05} test_acc={test_acc:.05}"
        )


if __name__ == "__main__":
    main()
