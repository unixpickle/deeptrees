import itertools

import torch
import torch.optim as optim
from deeptrees.analysis import randomize_tree_decisions, track_tree_usage
from deeptrees.cascade import Batch, CascadeSGD
from deeptrees.experiments.classifier.models import (
    conv_pool_soft_tree_small as model_initializer,
)
from deeptrees.experiments.data import load_mnist
from deeptrees.gradient_boosting import BoostingSoftmaxLoss

VERBOSE = False


def main():
    print("loading data...")
    xs, ys = load_mnist(train=True, spatial=True)
    test_xs, test_ys = load_mnist(train=False, spatial=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys = xs.to(device), ys.to(device)
    test_xs, test_ys = test_xs.to(device), test_ys.to(device)

    train_batch_size = 512
    outer_batch_size = 1024

    print("initializing TAO model...")
    init_batch_size = 2048
    model, _ = model_initializer()(Batch.with_x(xs[:init_batch_size]))
    sgd_model = CascadeSGD(
        model,
        opt=optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1),
        schedule=["sgd"] * 15 + ["base"] * 15,
    )
    print(f"model has {sum(x.numel() for x in model.parameters())} parameters.")

    print("training...")
    loss = BoostingSoftmaxLoss()
    for epoch in itertools.count():
        sgd_model.update(
            full_batch=Batch.with_x(xs),
            loss_fn=lambda indices, batch: loss(batch.x, ys[indices]),
            batch_size=train_batch_size,
            outer_batch_size=outer_batch_size,
        )
        test_loss, test_acc = compute_loss_acc(sgd_model, test_xs, test_ys, loss)
        with track_tree_usage(sgd_model) as tree_usage:
            train_loss, train_acc = compute_loss_acc(sgd_model, xs, ys, loss)
        with randomize_tree_decisions(sgd_model):
            _, train_acc_rand = compute_loss_acc(sgd_model, xs, ys, loss)
        print(
            f"epoch {epoch}: train_loss={train_loss:.05} train_acc={train_acc:.05} train_acc_rand={train_acc_rand:.05} test_loss={test_loss:.05} test_acc={test_acc:.05}"
        )
        if VERBOSE:
            for tree, usage in tree_usage.items():
                print(
                    f"  - tree {tree}: used={usage.num_used()} entropy={usage.entropy():.05}"
                )


@torch.no_grad()
def compute_loss_acc(model, xs, ys, loss_fn, batch_size=1024):
    model.eval()
    total_test_loss = 0.0
    total_test_correct = 0.0
    for i in range(0, len(xs), batch_size):
        test_out = model(Batch.with_x(xs[i : i + batch_size])).x
        test_losses = loss_fn(test_out, ys[i : i + batch_size])
        total_test_correct += (
            (test_out.argmax(-1) == ys[i : i + batch_size]).long().sum().item()
        )
        total_test_loss += test_losses.sum().item()
    test_loss = total_test_loss / len(xs)
    test_acc = total_test_correct / len(xs)
    model.train()
    return test_loss, test_acc


if __name__ == "__main__":
    main()
