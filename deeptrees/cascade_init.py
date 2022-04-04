import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor

from .cascade import (
    Batch,
    CascadeCheckpoint,
    CascadeConv,
    CascadeGradientLoss,
    CascadeLinearGatedTAO,
    CascadeModule,
    CascadeSequential,
    CascadeTAO,
    extract_image_patches,
    flatten_image_patches,
    undo_image_patches,
)
from .cascade_nvp import (
    CascadeNVPCheckpoint,
    CascadeNVPGradientLoss,
    CascadeNVPPartial,
    CascadeNVPSequential,
)
from .fit_base import TreeBranchBuilder
from .fit_sklearn import SklearnLinearLeafBuilder, SklearnRegressionTreeBuilder
from .fit_torch import TorchObliqueBranchBuilder
from .tao import TAOTreeBuilder
from .tree import (
    AxisTreeBranch,
    ConstantTreeLeaf,
    LinearTreeLeaf,
    ObliqueTreeBranch,
    Tree,
)


@dataclass
class CascadeInit(ABC):
    @abstractmethod
    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        """
        Initialize a module for the current input batch.

        :param inputs: some sample inputs to the layer.
        :param targets: (possibly unused) batch of targets for initializing a
                        model grounded in real data.
        :return: a tuple (layer, outputs).
        """


@dataclass
class CascadeRawInit(CascadeInit):
    module: CascadeModule

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        with torch.no_grad():
            inputs = self.module(inputs)
        _ = targets
        return self.module, inputs


@dataclass
class CascadeTAOInit(CascadeInit):
    """
    Randomly initialize a CascadeTAO layer by repeatedly halving the training
    data along random axes.
    """

    out_size: int
    tree_depth: int
    branch_builder: TreeBranchBuilder
    reject_unimprovement: bool = True
    random_prob: float = 0.0
    zero_init_out: bool = False

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        _ = targets
        tree = random_tree(
            inputs.x,
            self.out_size,
            self.tree_depth,
            random_prob=self.random_prob,
            zero_init_out=self.zero_init_out,
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
class CascadeTAOTreeBuilderInit(CascadeInit):
    """
    Initialize a CascadeTAO layer using a TAOTreeBuilder to initialize the tree
    for an initial batch of data.
    """

    builder: TAOTreeBuilder
    out_size: int
    random_prob: float = 0.0

    @classmethod
    def regression_init_builder(
        cls,
        depth: int,
        out_size: int,
        random_prob: float = 0.0,
        branch_builder_max_epochs: int = 50,
        reject_unimprovement: bool = True,
        max_iterations: int = 1,
        **tao_kwargs,
    ):
        return cls(
            builder=TAOTreeBuilder(
                loss_fn=lambda x, y: ((x - y) ** 2).flatten(1).mean(1),
                base_builder=SklearnRegressionTreeBuilder(
                    estimator=DecisionTreeRegressor(max_depth=depth)
                ),
                leaf_builder=SklearnLinearLeafBuilder(),
                branch_builder=TorchObliqueBranchBuilder(
                    max_epochs=branch_builder_max_epochs
                ),
                reject_unimprovement=reject_unimprovement,
                max_iterations=max_iterations,
                **tao_kwargs,
            ),
            out_size=out_size,
            random_prob=random_prob,
        )

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        assert targets is not None
        tree = (
            self.builder.fit(inputs.x, targets.x)
            .map_leaves(
                lambda _: random_tree(
                    xs=inputs.x,
                    out_size=self.out_size,
                    depth=0,
                    random_prob=self.random_prob,
                )
            )
            .map_branches(
                lambda x: (
                    x
                    if not isinstance(x, AxisTreeBranch)
                    else x.to_oblique(inputs.x, self.random_prob)
                )
            )
        )

        def apply_fn(module):
            if isinstance(module, ObliqueTreeBranch):
                module.random_prob = self.random_prob

        tree.apply(apply_fn)
        return (
            CascadeTAO(
                tree=tree,
                branch_builder=self.builder.branch_builder,
                reject_unimprovement=self.builder.reject_unimprovement,
            ),
            Batch.with_x(tree(inputs.x)),
        )


@dataclass
class CascadeLinearGatedTAOInit(CascadeTAOInit):
    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        _ = targets
        tree = random_tree(
            inputs.x,
            self.out_size,
            self.tree_depth,
            random_prob=self.random_prob,
            constant_leaf=True,
        )
        in_size = inputs.x.shape[1]
        layer = nn.Linear(in_size, self.out_size).to(inputs.x)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(self.out_size, in_size) / math.sqrt(in_size))
            layer.bias.zero_()
            gates = tree(inputs.x).tanh() + 1
            inputs = Batch.with_x(layer(inputs.x) * gates)
        return (
            CascadeLinearGatedTAO(
                tree,
                linear_layer=layer,
                branch_builder=self.branch_builder,
                reject_unimprovement=self.reject_unimprovement,
            ),
            inputs,
        )


@dataclass
class CascadeNVPPartialInit(CascadeInit):
    initializer: CascadeInit
    learn_scale: bool = False

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        _ = targets
        in_size = inputs.x.shape[1]
        assert in_size % 2 == 0, "must operate on an even number of features"

        sep = torch.zeros(in_size, dtype=torch.bool, device=inputs.x.device)
        sep[torch.randperm(in_size, device=sep.device)[: in_size // 2]] = True

        result = []
        for mask in [sep, ~sep]:
            wrapped_layer, _ = self.initializer(
                Batch.with_x(inputs.x[:, mask]), Batch.with_x(inputs.x[:, ~mask])
            )
            layer = CascadeNVPPartial(mask, wrapped_layer, learn_scale=self.learn_scale)
            result.append(layer)
            with torch.no_grad():
                inputs = layer(inputs)
        return CascadeNVPSequential(result), inputs


@dataclass
class CascadeTAONVPInit(CascadeInit):
    tree_depth: int
    branch_builder: TreeBranchBuilder
    reject_unimprovement: bool = True
    random_prob: float = 0.0
    regression_init: bool = False
    learn_scale: bool = False

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        _ = targets
        in_size = inputs.x.shape[1]
        assert in_size % 2 == 0, "must operate on an even number of features"

        sep = torch.zeros(in_size, dtype=torch.bool, device=inputs.x.device)
        sep[torch.randperm(in_size, device=sep.device)[: in_size // 2]] = True

        result = []
        for mask in [sep, ~sep]:
            out_size = 2 * (~mask).long().sum().item()
            if self.regression_init:
                tree = SklearnRegressionTreeBuilder(
                    estimator=DecisionTreeRegressor(max_depth=self.tree_depth),
                ).fit(inputs.x[:, mask], inputs.x[:, ~mask])
                zero_out = torch.zeros(
                    out_size,
                    dtype=inputs.x.dtype,
                    device=inputs.x.device,
                )
                tree = tree.map_branches(
                    lambda x: x.to_oblique(
                        inputs.x[:, mask], random_prob=self.random_prob
                    )
                ).map_leaves(lambda _: ConstantTreeLeaf(zero_out))
            else:
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
                learn_scale=self.learn_scale,
            )
            result.append(layer)
            with torch.no_grad():
                inputs = layer(inputs)
        return CascadeNVPSequential(result), inputs


@dataclass
class CascadeGradientLossInit(CascadeInit):
    contained: CascadeInit
    damping: float = 0.0
    sign_only: bool = False
    nvp: bool = False

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        module, outs = self.contained(inputs, targets)
        wrapper = CascadeNVPGradientLoss if self.nvp else CascadeGradientLoss
        return wrapper(module, damping=self.damping, sign_only=self.sign_only), outs


@dataclass
class CascadeCheckpointInit(CascadeInit):
    contained: CascadeInit
    nvp: bool = False

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        module, outs = self.contained(inputs, targets)
        wrapper = CascadeNVPCheckpoint if self.nvp else CascadeCheckpoint
        return wrapper(module), outs


@dataclass
class CascadeSequentialInit(CascadeInit):
    initializers: Sequence[CascadeInit]
    nvp: bool = False
    flatten: bool = False

    @classmethod
    def tao_dense(
        cls,
        hidden_sizes: Sequence[int],
        tree_depth: int,
        branch_builder: TreeBranchBuilder,
        random_prob: float = 0.0,
        reject_unimprovement: bool = True,
        zero_init_out: bool = False,
    ):
        return cls(
            [
                CascadeTAOInit(
                    out_size=x,
                    tree_depth=tree_depth,
                    branch_builder=branch_builder,
                    random_prob=random_prob,
                    reject_unimprovement=reject_unimprovement,
                    zero_init_out=zero_init_out and i + 1 == len(hidden_sizes),
                )
                for i, x in enumerate(hidden_sizes)
            ]
        )

    @classmethod
    def linear_gated_tao(
        cls,
        hidden_sizes: Sequence[int],
        tree_depth: int,
        branch_builder: TreeBranchBuilder,
        random_prob: float = 0.0,
        reject_unimprovement: bool = True,
    ):
        return cls(
            [
                CascadeLinearGatedTAOInit(
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
        regression_init: bool = False,
        learn_scale: bool = False,
    ):
        assert num_layers % 2 == 0, "must have even number of layers"
        return cls(
            [
                CascadeTAONVPInit(
                    tree_depth=tree_depth,
                    branch_builder=branch_builder,
                    random_prob=random_prob,
                    reject_unimprovement=reject_unimprovement,
                    regression_init=regression_init,
                    learn_scale=learn_scale,
                )
                for _ in range(num_layers // 2)
            ],
            nvp=True,
            flatten=True,
        )

    def map(self, fn: Callable[[CascadeInit], CascadeInit]) -> "CascadeSequentialInit":
        return CascadeSequentialInit(
            initializers=[fn(x) for x in self.initializers],
            nvp=self.nvp,
        )

    def checkpoint(self, chunk_size: Optional[int] = None) -> "CascadeSequentialInit":
        """
        Wrap sub-sequences of modules with CascadeCheckpointInit.

        :param chunk_size: the number of modules per wrapped sequence. If not
                           specified, the sqrt() of all modules is used.
        """
        if chunk_size is None:
            chunk_size = round(math.sqrt(len(self.initializers)))
        if not chunk_size or chunk_size == len(self.initializers):
            return self
        results = []
        for i in range(0, len(self.initializers), chunk_size):
            results.append(
                CascadeCheckpointInit(
                    CascadeSequentialInit(
                        self.initializers[i : i + chunk_size], nvp=self.nvp
                    ),
                    nvp=self.nvp,
                )
            )
        return CascadeSequentialInit(results, nvp=self.nvp)

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        result = []
        for x in self.initializers:
            module, inputs = x(inputs, targets)
            if self.flatten and isinstance(module, CascadeSequential):
                result.extend(module)
            else:
                result.append(module)
        return (CascadeSequential if not self.nvp else CascadeNVPSequential)(
            result
        ), inputs


@dataclass
class CascadeConvInit(CascadeInit):
    contained: CascadeInit
    kernel_size: int
    stride: int
    padding: int
    subsampling: Optional[float] = None
    gradient_loss: bool = False

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        x = extract_image_patches(
            inputs.x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        contained, outputs = self.contained(
            inputs.with_x(flatten_image_patches(x)), targets
        )
        result_module = CascadeConv(
            contained=contained,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            subsampling=self.subsampling,
            gradient_loss=self.gradient_loss,
        )
        inputs = inputs.with_x(undo_image_patches(x.shape, outputs.x))

        return result_module, inputs


def random_tree(
    xs: torch.Tensor,
    out_size: int,
    depth: int,
    random_prob: float = 0.0,
    constant_leaf: bool = False,
    zero_init_out: bool = False,
) -> Tree:
    in_size = xs.shape[1]
    if depth == 0:
        if constant_leaf:
            return ConstantTreeLeaf(torch.zeros(out_size).to(xs))
        else:
            return LinearTreeLeaf(
                coef=(
                    torch.randn(size=(in_size, out_size)).to(xs) / math.sqrt(in_size)
                    if not zero_init_out
                    else torch.zeros(in_size, out_size).to(xs)
                ),
                bias=torch.zeros(out_size).to(xs),
            )
    split_direction = torch.randn(in_size).to(xs)
    dots = (xs @ split_direction).view(-1)
    threshold = dots.median()
    decision = dots > threshold
    return ObliqueTreeBranch(
        left=random_tree(
            xs[~decision],
            out_size,
            depth - 1,
            constant_leaf=constant_leaf,
            zero_init_out=zero_init_out,
        ),
        right=random_tree(
            xs[decision],
            out_size,
            depth - 1,
            constant_leaf=constant_leaf,
            zero_init_out=zero_init_out,
        ),
        coef=split_direction,
        threshold=threshold,
        random_prob=random_prob,
    )
