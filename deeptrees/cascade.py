from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from .fit_base import TreeBranchBuilder
from .tao import TAOBase, TAOResult
from .tree import Tree, TreeBranch, TreeLeaf


class Batch(dict):
    """
    A dict of tensors where each tensor should have the same length along the
    first axis.
    """

    @classmethod
    def with_x(cls, x: torch.Tensor) -> "Batch":
        return cls(dict(x=x))

    @classmethod
    def cat(cls, elements: Sequence["Batch"], dim: int = 0) -> "Batch":
        joined = defaultdict(list)
        for element in elements:
            for k, v in element.items():
                joined[k].append(v)
        return cls({k: torch.cat(v, dim=dim) for k, v in joined.items()})

    @property
    def x(self) -> torch.Tensor:
        return self["x"]

    def detach(self) -> "Batch":
        return Batch({k: v.detach() for k, v in self.items()})

    def clone(self) -> "Batch":
        return Batch({k: v.clone() for k, v in self.items()})

    def at_indices(self, indices: torch.Tensor) -> "Batch":
        return Batch({k: v[indices] for k, v in self.items()})

    def force_requires_grad(self) -> "Batch":
        return Batch(
            {
                k: v if v.requires_grad else v.detach().requires_grad_()
                for k, v in self.items()
            }
        )

    def batches(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, "Batch"]]:
        size = len(self)
        indices = torch.arange(size)
        for i in range(0, size, batch_size):
            yield indices[i : i + batch_size], Batch(
                {k: v[i : i + batch_size] for k, v in self.items()}
            )

    def __len__(self) -> int:
        return len(next(iter(self.values())))


BatchLossFn = Callable[[torch.Tensor, Batch], torch.Tensor]


class UpdateContext:
    def __init__(self):
        self._inputs_cache: Dict[nn.Module, List[Batch]] = defaultdict(list)
        self._outputs_cache: Dict[nn.Module, List[Batch]] = defaultdict(list)
        self._grads_cache: Dict[nn.Module, List[Batch]] = defaultdict(list)
        self._cur_autograd_cache: Dict[nn.Module, Batch] = dict()
        self._loss_cache: List[torch.Tensor] = []

    def cache_inputs(self, module: nn.Module, inputs: Batch):
        """
        Call during module forward() to cache a batch of inputs.
        """
        self._inputs_cache[module].append(inputs.detach())

    def cache_outputs(self, module: nn.Module, outputs: Batch):
        """
        Call during module forward() to cache a batch of outputs.
        """
        self._outputs_cache[module].append(outputs.detach())

    def require_grad(self, module: nn.Module, outputs: Batch) -> Batch:
        """
        Call during module forward() to request a gradient for the outputs.
        Returns a new Batch that should be returned from the module.
        """
        assert module not in self._cur_autograd_cache
        outputs = outputs.force_requires_grad()
        self._cur_autograd_cache[module] = outputs
        return outputs

    def backward(self, losses: torch.Tensor):
        """
        Update the gradient cache for batches passed to require_grad() based on
        the loss.
        """
        self._loss_cache.append(losses.detach())
        tensors = []
        for batch in self._cur_autograd_cache.values():
            tensors.extend(x for x in batch.values() if x.requires_grad)
        if len(tensors) == 0:
            self._cur_autograd_cache.clear()
            return
        grads = torch.autograd.grad(losses.sum(), tensors, allow_unused=True)
        grads = list(grads)

        for module, batch in self._cur_autograd_cache.items():
            grad_batch = Batch()
            for k, v in batch.items():
                if v.requires_grad:
                    next_grad = grads.pop(0)
                    if next_grad is not None:
                        grad_batch[k] = next_grad
            self._grads_cache[module].append(grad_batch)
        assert not len(grads), "did not consume all gradients while structuring them"

        self._cur_autograd_cache.clear()

    def get_losses(self) -> torch.Tensor:
        """
        Get all of the losses concatenated from all of the backward() calls.
        """
        return torch.cat(self._loss_cache, dim=0)

    def get_inputs(
        self, module: nn.Module, concatenate: bool = True, remove: bool = True
    ) -> Union[Batch, List[Batch]]:
        """
        Get all of the cached inputs for the given module.
        """
        return self._get_batch(self._inputs_cache, module, concatenate, remove)

    def get_outputs(
        self,
        module: nn.Module,
        concatenate: bool = True,
        remove: bool = True,
    ) -> Union[Batch, List[Batch]]:
        """
        Get all of the cached outputs for the given module.
        """
        return self._get_batch(self._outputs_cache, module, concatenate, remove)

    def get_grads(
        self, module: nn.Module, concatenate: bool = True, remove: bool = True
    ) -> Union[Batch, List[Batch]]:
        """
        Get all of the cached grads for the given module.
        """
        return self._get_batch(self._grads_cache, module, concatenate, remove)

    def _get_batch(
        self,
        batch: Dict[nn.Module, List[Batch]],
        module: nn.Module,
        concatenate: bool,
        remove: bool,
    ) -> Union[Batch, List[Batch]]:
        if concatenate:
            assert module in batch, f"module has no recorded values"
        results = batch[module]
        if remove:
            del batch[module]
        if not concatenate:
            return results
        else:
            return Batch.cat(results, dim=0)


class CascadeModule(nn.Module, ABC):
    """
    An CascadeModule is a learnable component of a model that can be used in
    a deep network of decision trees. Unlike regular deep learning modules,
    each cascade module gets a loss function that determines, per input, what
    the global loss *would be* given an arbitrary output vector.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs: Batch, ctx: Optional[UpdateContext] = None) -> Batch:
        """
        Run the module, caching values in ctx if necessary.
        """

    @abstractmethod
    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        """
        Update the parameters of the module and its submodules given the loss
        function and context that was previously used for a forward pass.
        """

    def update(
        self, full_batch: Batch, loss_fn: BatchLossFn, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Update the parameters of the module and all its submodules given the
        inputs and loss function. Gradients will automatically be calculated
        as necessary, and update_local() will be called on self.

        This should be called once on the root module, since it will amortize
        the cost of running backpropagation throughout the model.

        :param full_batch: a large batch of training examples.
        :param loss_fn: the global loss function.
        :param batch_size: if specified, split the batch up into minibatches
                           during the initial forward/backward passes.
        :return: a 1-D tensor of pre-update losses.
        """
        ctx = UpdateContext()
        for indices, sub_batch in full_batch.batches(batch_size or len(full_batch)):
            losses = loss_fn(indices, self(sub_batch, ctx=ctx))
            ctx.backward(losses)
            del losses  # possibly save memory if backward() didn't destroy graph.
        self.update_local(ctx, loss_fn)
        return ctx.get_losses()


class CascadeSequential(CascadeModule):
    """
    Sequentially compose multiple cascade modules.
    """

    def __init__(self, sequence: Sequence[CascadeModule]):
        super().__init__()
        self.sequence = nn.ModuleList(sequence)

    def forward(self, inputs: Batch, ctx: Optional[UpdateContext] = None) -> Batch:
        out = inputs
        for layer in self.sequence:
            out = layer(out, ctx=ctx)
        return out

    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        for i in range(len(self.sequence))[::-1]:

            def child_loss_fn(indices: torch.Tensor, outputs: Batch) -> torch.Tensor:
                with torch.no_grad():
                    final_outputs = outputs
                    for layer in self.sequence[i + 1 :]:
                        final_outputs = layer(final_outputs)
                    return loss_fn(indices, final_outputs)

            self.sequence[i].update_local(ctx, child_loss_fn)


class CascadeTAO(CascadeModule):
    """
    A tree in a cascade module that uses TAO for updates.

    The provided TreeBranchBuilder should not change the types or structure of
    the tree. In particular, the state_dict of trees should not change keys or
    the shapes of values across updates.
    """

    def __init__(
        self,
        tree: Tree,
        branch_builder: TreeBranchBuilder,
        reject_unimprovement: bool = True,
    ):
        super().__init__()
        self.tree = tree
        self.branch_builder = branch_builder
        self.reject_unimprovement = reject_unimprovement

    def forward(self, inputs: Batch, ctx: Optional[UpdateContext] = None) -> Batch:
        if ctx is not None:
            ctx.cache_inputs(self, inputs)
        return Batch.with_x(self.tree(inputs.x))

    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        h_tao = _CascadeTAO(
            xs=ctx.get_inputs(self).x,
            loss_fn=loss_fn,
            branch_builder=self.branch_builder,
            reject_unimprovement=self.reject_unimprovement,
        )
        self.tree.load_state_dict(h_tao.optimize(self.tree).tree.state_dict())


class CascadeLinearGatedTAO(CascadeModule):
    """
    A linear layer paired with a tree that outputs gates for the outputs of the
    linear layer.

    The provided TreeBranchBuilder should not change the types or structure of
    the tree. In particular, the state_dict of trees should not change keys or
    the shapes of values across updates.
    """

    def __init__(
        self,
        tree: Tree,
        linear_layer: nn.Linear,
        branch_builder: TreeBranchBuilder,
        reject_unimprovement: bool = True,
    ):
        super().__init__()
        self.tree = tree
        self.linear_layer = linear_layer
        self.branch_builder = branch_builder
        self.reject_unimprovement = reject_unimprovement

    def forward(self, inputs: Batch, ctx: Optional[UpdateContext] = None) -> Batch:
        if ctx is not None:
            ctx.cache_inputs(self, inputs)
        gates = self.tree(inputs.x).tanh() + 1
        return Batch.with_x(gates * self.linear_layer(inputs.x))

    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        inputs = ctx.get_inputs(self)

        with torch.no_grad():
            linear_outputs = self.linear_layer(inputs.x)

        def tao_loss_fn(indices: torch.Tensor, batch: Batch) -> torch.Tensor:
            return loss_fn(
                indices, Batch.with_x((batch.x.tanh() + 1) * linear_outputs[indices])
            )

        h_tao = _CascadeTAO(
            xs=inputs.x,
            loss_fn=tao_loss_fn,
            branch_builder=self.branch_builder,
            reject_unimprovement=self.reject_unimprovement,
        )
        self.tree.load_state_dict(h_tao.optimize(self.tree).tree.state_dict())


@dataclass
class _CascadeTAO(TAOBase):
    """
    A concrete subclass of TAOBase that implements TAO with leaves that get
    passed to the remainder of a network.
    """

    loss_fn: BatchLossFn
    branch_builder: TreeBranchBuilder
    reject_unimprovement: bool

    def build_branch(
        self,
        cur_branch: TreeBranch,
        cur_decision: torch.Tensor,
        sample_indices: torch.Tensor,
        left_losses: torch.Tensor,
        right_losses: torch.Tensor,
    ) -> TAOResult:
        xs = self.xs[sample_indices]
        tree = self.branch_builder.fit_branch(
            cur_branch=cur_branch,
            xs=xs,
            left_losses=left_losses,
            right_losses=right_losses,
        )
        with torch.no_grad():
            losses = torch.where(tree.decision(xs), right_losses, left_losses)
            if self.reject_unimprovement:
                old_losses = torch.where(cur_decision, right_losses, left_losses)
                if old_losses.mean().item() < losses.mean().item():
                    tree = cur_branch
                    losses = old_losses
        return TAOResult(tree=tree, losses=losses)

    def build_leaf(self, cur_leaf: TreeLeaf, sample_indices: torch.Tensor) -> TAOResult:
        outputs = cur_leaf(self.xs[sample_indices])
        return TAOResult(
            tree=cur_leaf, losses=self.output_loss(sample_indices, outputs)
        )

    def output_loss(
        self, sample_indices: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(sample_indices, Batch.with_x(outputs))


class CascadeGradientLoss(CascadeModule):
    """
    Wrap a module to use a local linear loss function based on a gradient.
    This prevents expensive forward propagation through the rest of the model.
    """

    def __init__(
        self, contained: CascadeModule, damping: float = 0.0, sign_only: bool = False
    ):
        super().__init__()
        self.contained = contained
        self.damping = damping
        self.sign_only = sign_only

    def forward(self, x: Batch, ctx: Optional[UpdateContext] = None):
        out = self.contained(x, ctx=ctx)
        if ctx is not None:
            ctx.cache_outputs(self, out)
            return ctx.require_grad(self, out)
        else:
            return out

    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        _ = loss_fn
        original_out = ctx.get_outputs(self)
        gradient = ctx.get_grads(self)
        assert len(gradient), "at least one output must have gradient info"

        def local_loss_fn(indices: torch.Tensor, outputs: Batch) -> torch.Tensor:
            losses = 0.0
            x0 = original_out.at_indices(indices)
            for k, g in gradient.at_indices(indices).items():
                if self.damping:
                    losses = losses + self.damping * (
                        (outputs[k] - x0[k]) ** 2
                    ).flatten(1).sum(1)
                dots = ((outputs[k] - x0[k]) * g).flatten(1).sum(1)
                if self.sign_only:
                    dots = dots.sign().float()
                losses = losses + dots
            return losses

        return self.contained.update_local(ctx, local_loss_fn)


class CascadeSGD(CascadeModule):
    """
    Wrap sub-modules to perform frequent gradient-based updates and less
    frequent recursive update() calls.
    """

    def __init__(self, contained: CascadeModule, interval: int, opt: optim.Optimizer):
        super().__init__()
        self.contained = contained
        self.interval = interval
        self.optimizer = opt
        params = list(contained.parameters())
        if len(params):
            device = params[0].device
        else:
            device = torch.device("cpu")
        self.register_buffer("step", torch.tensor(0, dtype=torch.int64, device=device))

    def forward(self, inputs: Batch, ctx: Optional[UpdateContext] = None) -> Batch:
        return self.contained(inputs, ctx=ctx)

    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        self.contained.update_local(ctx, loss_fn)

    def update(
        self, full_batch: Batch, loss_fn: BatchLossFn, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        self.step.add_(1)
        if self.step.item() % self.interval == 0:
            return super().update(full_batch, loss_fn, batch_size=batch_size)
        else:
            self.optimizer.zero_grad()
            all_losses = []
            for indices, sub_batch in full_batch.batches(batch_size or len(full_batch)):
                losses = loss_fn(indices, self(sub_batch))
                all_losses.append(losses.detach())
                losses.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            return torch.cat(all_losses, dim=0)
