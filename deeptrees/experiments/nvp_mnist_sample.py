import argparse

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="samples.png")
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        model = torch.load(f, map_location="cpu")

    d = 28
    samples = model.invert(torch.randn(args.grid_size ** 2, d ** 2), [])
    samples = (samples * 255).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    samples = samples.reshape([args.grid_size, args.grid_size, d, d, 1])
    samples = samples.transpose(0, 2, 1, 3, 4)
    samples = samples.reshape([args.grid_size * d, args.grid_size * d, 1])
    Image.fromarray(np.tile(samples, [1, 1, 3])).save(args.output)


if __name__ == "__main__":
    main()
