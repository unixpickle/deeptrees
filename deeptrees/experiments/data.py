import os
import warnings
from typing import Tuple, Union

import numpy as np
import torch
from torchvision.datasets.mnist import MNIST, FashionMNIST

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_mnist(
    train: bool, fashion: bool = False, spatial: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        dataset = (MNIST if not fashion else FashionMNIST)(
            os.path.join(DATA_DIR, "mnist" if not fashion else "fashion_mnist"),
            train=train,
            download=True,
        )
        return dataset_to_tensors(dataset, spatial=spatial)


def dataset_to_tensors(
    dataset: Union[MNIST, FashionMNIST], spatial=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    shape = (-1,) if not spatial else (1, 28, 28)
    for i in range(len(dataset)):
        img, target = dataset[i]
        images.append(
            torch.from_numpy(np.array(img.convert("RGB"))[..., 0]).view(shape).float()
            / 255
        )
        labels.append(target)
    return torch.stack(images, dim=0), torch.tensor(labels)
