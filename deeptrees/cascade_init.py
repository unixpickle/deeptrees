import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import torch

from .cascade import Batch, CascadeModule, CascadeSequential, CascadeTAO
from .cascade_nvp import CascadeNVPPartial, CascadeNVPSequential
from .fit_base import TreeBranchBuilder
from .tree import ConstantTreeLeaf, LinearTreeLeaf, ObliqueTreeBranch, Tree


@dataclass
class CascadeInit(ABC):
    @abstractmethod
    def __call__(
        self, inputs: Batch
    ) -> Tuple[Union[CascadeModule, List[CascadeModule]], Batch]:
        """
        Initialize a sequence of modules for the current input batch.

        :param inputs: some sample inputs to the layer.
        :return: a tuple (layers, outputs).
        """


@dataclass
class CascadeTAOInit(CascadeInit):
    out_size: int
    tree_depth: int
    branch_builder: TreeBranchBuilder
    reject_unimprovement: bool = True
    random_prob: float = 0.0

    def __call__(
        self, inputs: Batch
    ) -> Tuple[Union[CascadeModule, List[CascadeModule]], Batch]:
        tree = random_tree(
            inputs.x, self.out_size, self.tree_depth, random_prob=self.random_prob
        )
        with torch.no_grad():
            inputs = Batch.with_x(tree(inputs.x))
        return (
            CascadeTAO(
                tree,
                branch_builder=self.branch_builder,
                reject_unimprovement=self.reject_unimprovement,
            ),
            inputs,
        )


@dataclass
class CascadeTAONVPInit(CascadeInit):
    tree_depth: int
    branch_builder: TreeBranchBuilder
    reject_unimprovement: bool = True
    random_prob: float = 0.0

    def __call__(
        self, inputs: Batch
    ) -> Tuple[Union[CascadeModule, List[CascadeModule]], Batch]:
        in_size = inputs.x.shape[1]
        assert in_size % 2 == 0, "must operate on an even number of features"

        sep = torch.zeros(in_size, dtype=torch.bool, device=inputs.x.device)
        sep[torch.randperm(in_size, device=sep.device)[: in_size // 2]] = True

        result = []
        for mask in [sep, ~sep]:
            out_size = 2 * (~mask).long().sum().item()
            tree = random_tree(
                inputs.x[:, mask],
                out_size,
                self.tree_depth,
                random_prob=self.random_prob,
                constant_leaf=True,
            )
            layer = CascadeNVPPartial(
                mask,
                CascadeTAO(
                    tree,
                    branch_builder=self.branch_builder,
                    reject_unimprovement=self.reject_unimprovement,
                ),
            )
            result.append(layer)
            with torch.no_grad():
                inputs = layer(inputs)
        return result, inputs


@dataclass
class CascadeSequentialInit(CascadeInit):
    initializers: Sequence[CascadeInit]
    nvp: bool = False

    @classmethod
    def tao_dense(
        cls,
        hidden_sizes: Sequence[int],
        tree_depth: int,
        branch_builder: TreeBranchBuilder,
        random_prob: float = 0.0,
        reject_unimprovement: bool = True,
    ):
        return cls(
            [
                CascadeTAOInit(
                    out_size=x,
                    tree_depth=tree_depth,
                    branch_builder=branch_builder,
                    random_prob=random_prob,
                    reject_unimprovement=reject_unimprovement,
                )
                for x in hidden_sizes
            ]
        )

    @classmethod
    def tao_nvp(
        cls,
        num_layers: int,
        tree_depth: int,
        branch_builder: TreeBranchBuilder,
        random_prob: float = 0.0,
        reject_unimprovement: bool = True,
    ):
        assert num_layers % 2 == 0, "must have even number of layers"
        return cls(
            [
                CascadeTAONVPInit(
                    tree_depth=tree_depth,
                    branch_builder=branch_builder,
                    random_prob=random_prob,
                    reject_unimprovement=reject_unimprovement,
                )
                for _ in range(num_layers // 2)
            ],
            nvp=True,
        )

    def __call__(
        self, inputs: Batch
    ) -> Tuple[Union[CascadeModule, List[CascadeModule]], Batch]:
        result = []
        for x in self.initializers:
            layers, inputs = x(inputs)
            if isinstance(layers, list):
                result.extend(layers)
            else:
                result.append(layers)
        return (CascadeSequential if not self.nvp else CascadeNVPSequential)(
            result
        ), inputs


def random_tree(
    xs: torch.Tensor,
    out_size: int,
    depth: int,
    random_prob: float = 0.0,
    constant_leaf: bool = False,
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
        random_prob=random_prob,
    )
