import math
from typing import Sequence

import torch

from .cascade import Batch, CascadeSequential, CascadeTAO
from .cascade_nvp import CascadeNVPPartial
from .fit_base import TreeBranchBuilder
from .tree import ConstantTreeLeaf, LinearTreeLeaf, ObliqueTreeBranch, Tree


def initialize_tao_dense(
    xs: torch.Tensor,
    hidden_sizes: Sequence[int],
    tree_depth: int,
    branch_builder: TreeBranchBuilder,
    **tao_kwargs,
) -> CascadeSequential:
    """
    Initialize a dense feedforward network with the given hidden vector sizes.
    The last hidden size should be the output dimensionality.
    """
    cur_data = xs
    layers = []
    for out_size in hidden_sizes:
        tree = random_tree(cur_data, out_size, tree_depth)
        cur_data = tree(cur_data)
        layers.append(CascadeTAO(tree, branch_builder=branch_builder, **tao_kwargs))
    return CascadeSequential(layers)


def initialize_tao_nvp(
    xs: torch.Tensor,
    num_layers: int,
    tree_depth: int,
    branch_builder: TreeBranchBuilder,
    **tao_kwargs,
) -> CascadeSequential:
    """
    Initialize a cascaded RealNVP-like model as a series of CascadeNVPPartial
    layers, where each layer contains a TAO module with constant output leaves.
    """
    in_size = xs.shape[1]
    assert in_size % 2 == 0, "must operate on an even number of features"
    assert (
        num_layers % 2 == 0
    ), "must have even number of layers for fair distribution of masks"
    cur_data = xs
    layers = []
    for _ in range(num_layers // 2):
        sep = torch.zeros(in_size, dtype=torch.bool, device=xs.device)
        sep[torch.randperm(in_size, device=sep.device)[: in_size // 2]] = True

        for mask in [sep, ~sep]:
            out_size = 2 * (~mask).long().sum().item()
            tree = random_tree(
                cur_data[:, mask], out_size, tree_depth, constant_leaf=True
            )
            layer = CascadeNVPPartial(
                mask, CascadeTAO(tree, branch_builder=branch_builder, **tao_kwargs)
            )
            layers.append(layer)
            cur_data = layer(Batch.with_x(cur_data)).x
    return CascadeSequential(layers)


def random_tree(
    xs: torch.Tensor, out_size: int, depth: int, constant_leaf: bool = False
) -> Tree:
    in_size = xs.shape[1]
    if depth == 0:
        if constant_leaf:
            return ConstantTreeLeaf(torch.zeros(out_size).to(xs))
        else:
            return LinearTreeLeaf(
                coef=torch.randn(size=(in_size, out_size)).to(xs) / math.sqrt(in_size),
                bias=torch.zeros(out_size).to(xs),
            )
    split_direction = torch.randn(in_size).to(xs)
    dots = (xs @ split_direction).view(-1)
    threshold = dots.median()
    decision = dots > threshold
    return ObliqueTreeBranch(
        left=random_tree(
            xs[~decision], out_size, depth - 1, constant_leaf=constant_leaf
        ),
        right=random_tree(
            xs[decision], out_size, depth - 1, constant_leaf=constant_leaf
        ),
        coef=split_direction,
        threshold=threshold,
    )
