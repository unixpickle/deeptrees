import itertools
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from deeptrees.cascade import Batch, CascadeSGD, CascadeTAO
from deeptrees.cascade_init import CascadeGradientLossInit, CascadeSequentialInit
from deeptrees.experiments.boosting_mnist import dataset_to_tensors
from deeptrees.fit_torch import TorchObliqueBranchBuilder
from deeptrees.gradient_boosting import BoostingSoftmaxLoss
from torchvision.datasets.mnist import MNIST

TIME_PER_RUN = 4 * 60
OUTPUT_DIR = "cascade_mnist_scan"
EARLY_STOP_EPOCHS = 50


def main():
    train_dataset = MNIST("./mnist_data", train=True, download=True)
    xs, ys = dataset_to_tensors(train_dataset)
    val_xs, val_ys = xs[:5000], ys[:5000]
    xs, ys = xs[5000:], ys[5000:]

    settings = list(
        itertools.product(
            [(64, 32, 10)],
            [2, 3],
            [0.2],
            [True],
            [False],
            [2, 5],
            [0.0, 0.1, 0.5],
            [0.0, 0.1, 1.0],
        )
    )
    field_names = [
        "hidden_sizes",
        "depth",
        "random_prob",
        "gradient_approx",
        "gated",
        "interval",
        "weight_decay",
        "leaf_variance_coef",
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for setting in settings:
        out_name = str(setting)
        out_path = os.path.join(OUTPUT_DIR, out_name + ".json")
        if os.path.exists(out_path):
            continue
        kwargs = dict(zip(field_names, setting))
        result = run_loss_curves(xs=xs, ys=ys, val_xs=val_xs, val_ys=val_ys, **kwargs)
        result["kwargs"] = kwargs
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"completed config {out_name}:")
        print(f" => epochs: {len(result['t'])}")
        print(f" => best val loss: {min(result['val_loss'])}")
        print(f" => best val loss epoch: {np.argmin(result['val_loss'])}")
        print(f" => best val acc: {max(result['val_acc'])}")
        print(f" => best val acc epoch: {np.argmax(result['val_acc'])}")
        print(f" => total time: {result['t'][-1]}")


def run_loss_curves(
    xs: torch.Tensor,
    ys: torch.Tensor,
    val_xs: torch.Tensor,
    val_ys: torch.Tensor,
    hidden_sizes: Tuple[int],
    depth: int,
    random_prob: float,
    gradient_approx: bool,
    gated: bool,
    interval: int,
    weight_decay: float,
    leaf_variance_coef: float,
) -> Dict[str, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys = xs.to(device), ys.to(device)
    val_xs, val_ys = val_xs.to(device), val_ys.to(device)

    initializer = (
        CascadeSequentialInit.tao_dense
        if not gated
        else CascadeSequentialInit.linear_gated_tao
    )(
        hidden_sizes=hidden_sizes,
        tree_depth=depth,
        branch_builder=TorchObliqueBranchBuilder(max_epochs=50),
        random_prob=random_prob,
    )
    if gradient_approx:
        initializer = initializer.map(CascadeGradientLossInit)

    model, _ = initializer(Batch.with_x(xs))
    sgd_model = CascadeSGD(
        model,
        interval=interval,
        opt=optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay),
    )

    loss = BoostingSoftmaxLoss()
    results = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], t=[])
    t0 = time.time()
    while time.time() < t0 + TIME_PER_RUN and not early_stop(results):
        _ = sgd_model.update(
            full_batch=Batch.with_x(xs),
            loss_fn=lambda indices, batch: (
                loss(batch.x, ys[indices])
                + leaf_variance_coef * leaf_variance(sgd_model)
            ),
            batch_size=1024,
        )
        with torch.no_grad():
            sgd_model.eval()
            train_out = sgd_model(Batch.with_x(xs)).x
            train_loss = loss(train_out, ys).mean().item()
            train_acc = (train_out.argmax(-1) == ys).float().mean().item()
            val_out = sgd_model(Batch.with_x(val_xs)).x
            val_loss = loss(val_out, val_ys).mean().item()
            val_acc = (val_out.argmax(-1) == val_ys).float().mean().item()
            sgd_model.train()
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)
        results["t"].append(time.time() - t0)
    return results


def early_stop(results: Dict[str, List[float]]) -> bool:
    acc = results["val_acc"]
    if not len(acc):
        return False
    best_it = np.argmax(acc)
    cur_it = len(acc) - 1
    return best_it + EARLY_STOP_EPOCHS < cur_it


def leaf_variance(model: CascadeSGD) -> torch.Tensor:
    result = [0.0]

    def apply_fn(module):
        if isinstance(module, CascadeTAO):
            result[0] = result[0] + module.leaf_variance()

    model.apply(apply_fn)
    return result[0]


if __name__ == "__main__":
    main()
