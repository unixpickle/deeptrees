import math
from typing import Sequence

import torch

from .cascade import CascadeSequential, CascadeTAO
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


def random_tree(
    xs: torch.Tensor, out_size: int, depth: int, constant_leaf: bool = False
) -> Tree:
    in_size = xs.shape[1]
    if depth == 0:
        if constant_leaf:
            return ConstantTreeLeaf(torch.randn(out_size).to(xs))
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
