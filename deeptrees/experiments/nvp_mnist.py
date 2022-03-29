import itertools
import math
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from deeptrees.cascade import Batch, CascadeSGD
from deeptrees.cascade_init import CascadeGradientLossInit, CascadeSequentialInit
from deeptrees.cascade_nvp import latents_from_batch, nvp_loss, quantization_noise
from deeptrees.experiments.boosting_mnist import dataset_to_tensors
from deeptrees.fit_torch import TorchObliqueBranchBuilder
from PIL import Image
from torchvision.datasets.mnist import MNIST

OUTPUT_DIR = "./models_nvp_mnist"
SAVE_INTERVAL = 10
GRID_SIZE = 32


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("loading data...")
    train_dataset = MNIST("./mnist_data", train=True, download=True)
    test_dataset = MNIST("./mnist_data", train=False, download=True)
    xs, ys = dataset_to_tensors(train_dataset)
    test_xs, test_ys = dataset_to_tensors(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys = xs.to(device), ys.to(device)
    test_xs, test_ys = test_xs.to(device), test_ys.to(device)

    print("initializing TAO model...")
    model, _ = (
        CascadeSequentialInit.tao_nvp(
            num_layers=96,
            tree_depth=4,
            branch_builder=TorchObliqueBranchBuilder(max_epochs=50),
            random_prob=0.0,
        )
        .map(lambda x: CascadeGradientLossInit(x, nvp=True))
        .checkpoint()(Batch.with_x(xs))
    )
    sgd_model = CascadeSGD(
        model, interval=5, opt=optim.Adam(model.parameters(), lr=3e-4)
    )
    print(f"model has {sum(x.numel() for x in model.parameters())} parameters.")

    print("training...")
    for epoch in itertools.count():
        losses = sgd_model.update(
            full_batch=Batch.with_x(quantization_noise(xs)),
            loss_fn=nvp_loss,
            batch_size=1024,
        )
        with torch.no_grad():
            sgd_model.eval()
            test_out = sgd_model(Batch.with_x(test_xs))
            test_losses = nvp_loss(None, test_out)
            sgd_model.train()
        print(
            f"epoch {epoch}: train_loss={losses.mean().item():.05} test_loss={test_losses.mean().item():.05}"
        )

        # Produce samples
        with torch.no_grad():
            sgd_model.eval()
            out_batch = model(Batch.with_x(test_xs[: GRID_SIZE ** 2]))
            latents = [torch.randn_like(x) for x in latents_from_batch(out_batch)]
            samples = model.invert(torch.randn_like(out_batch.x), latents)
            sgd_model.train()
            samples = (
                (samples * 255).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            )
            d = round(math.sqrt(samples.shape[1]))
            samples = samples.reshape([GRID_SIZE, GRID_SIZE, d, d, 1])
            samples = samples.transpose(0, 2, 1, 3, 4)
            samples = samples.reshape([GRID_SIZE * d, GRID_SIZE * d, 1])
            Image.fromarray(np.tile(samples, [1, 1, 3])).save(
                os.path.join(OUTPUT_DIR, "samples.png")
            )

        if (epoch + 1) % SAVE_INTERVAL == 0:
            print("saving model...")
            torch.save(model, os.path.join(OUTPUT_DIR, "model.pkl"))


if __name__ == "__main__":
    main()
