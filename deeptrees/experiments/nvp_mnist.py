import itertools
import os
import pickle

import torch
import torch.optim as optim
from deeptrees.cascade import Batch, CascadeSGD
from deeptrees.cascade_init import initialize_tao_nvp
from deeptrees.cascade_nvp import nvp_loss, quantization_noise
from deeptrees.experiments.boosting_mnist import dataset_to_tensors
from deeptrees.fit_torch import TorchObliqueBranchBuilder
from torchvision.datasets.mnist import MNIST

OUTPUT_DIR = "./models_nvp_mnist"
SAVE_INTERVAL = 10


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
    model = initialize_tao_nvp(
        xs=xs,
        num_layers=10,
        tree_depth=3,
        branch_builder=TorchObliqueBranchBuilder(max_epochs=50),
    )
    sgd_model = CascadeSGD(
        model, interval=5, opt=optim.Adam(model.parameters(), lr=1e-3)
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
            test_out = sgd_model(Batch.with_x(test_xs))
            test_losses = nvp_loss(None, test_out)
        print(
            f"epoch {epoch}: train_loss={losses.mean().item():.05} test_loss={test_losses.mean().item():.05}"
        )
        if (epoch + 1) % SAVE_INTERVAL == 0:
            with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
                pickle.dump(model, f)


if __name__ == "__main__":
    main()
