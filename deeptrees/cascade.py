from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .fit_base import TreeBranchBuilder
from .tao import TAOBase, TAOResult
from .tree import Tree, TreeBranch, TreeLeaf


@dataclass
class Batch:
    """
    A dict of tensors where each tensor should have the same length along the
    first axis.
    """

    data: Dict[str, torch.Tensor]

    @classmethod
    def with_x(cls, x: torch.Tensor) -> "Batch":
        return cls(dict(x=x))

    @classmethod
    def cat(cls, elements: Sequence["Batch"], dim: int = 0) -> "Batch":
        joined = defaultdict(list)
        for element in elements:
            for k, v in element.data.items():
                joined[k].append(v)
        return cls({k: torch.cat(v, dim=dim) for k, v in joined.items()})

    @property
    def x(self) -> torch.Tensor:
        return self.data["x"]

    def detach(self) -> "Batch":
        return Batch({k: v.detach() for k, v in self.data.items()})

    def clone(self) -> "Batch":
        return Batch({k: v.clone() for k, v in self.data.items()})

    def force_requires_grad(self) -> "Batch":
        return Batch(
            {
                k: v if v.requires_grad else v.detach().requires_grad_()
                for k, v in self.data.items()
            }
        )

    def batches(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, "Batch"]]:
        size = len(self)
        indices = torch.arange(size)
        for i in range(0, size, batch_size):
            yield indices[i : i + batch_size], Batch(
                {k: v[i : i + batch_size] for k, v in self.data.items()}
            )

    def __len__(self) -> int:
        return len(next(iter(self.data.values())))


BatchLossFn = Callable[[torch.Tensor, Batch], torch.Tensor]


@dataclass
class UpdateContext:
    inputs: Batch

    # This function takes two arguments: (indices, outputs).
    # The indices argument specifies which inputs the outputs correspond to in
    # the full batch.
    loss_fn: BatchLossFn

    # Only specified if a module's "require-_output_grad" returns true.
    outputs: Optional[Batch] = None
    output_grads: Optional[Batch] = None


class CascadeModule(nn.Module, ABC):
    """
    An CascadeModule is a learnable component of a model that can be used in
    a deep network of decision trees. Unlike regular deep learning modules,
    each cascade module gets a loss function that determines, per input, what
    the global loss *would be* given an arbitrary output vector.
    """

    def __init__(self):
        super().__init__()
        self._preparing_for_update = False
        self._cached_inputs = []
        self._cached_outputs = []
        self._cached_grads = []

    def forward(self, inputs: Batch) -> Batch:
        out = self.evaluate(inputs)

        if self._preparing_for_update:
            self._cached_inputs.append(inputs)

        if not (self._preparing_for_update and self.requires_output_grad()):
            return out

        self._cached_outputs.append(out.detach().clone())
        out = out.force_requires_grad()

        class _CacheGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(_ctx, *inputs):
                return inputs

            @staticmethod
            def backward(_ctx, *grad_outputs):
                self._cached_grads.append(
                    Batch(
                        {
                            k: v.detach().clone()
                            for k, v in zip(out.keys(), grad_outputs)
                        }
                    )
                )
                return grad_outputs

        return Batch(
            dict(zip(out.data.keys(), _CacheGradFunc.apply(out.data.values())))
        )

    def requires_output_grad(self) -> bool:
        """
        If this method returns True, calls to update() will include the output
        and output gradient of the loss function for this module.
        """

    @abstractmethod
    def evaluate(self, inputs: Batch) -> Batch:
        """
        Evaluate the output of the module given the input.

        Subclasses should override this method instead of forward().
        """

    @abstractmethod
    def update_local(self, ctx: UpdateContext):
        """
        Update the parameters of the module and its submodules given the
        inputs, loss function, and possibly gradients of the loss function
        w.r.t. the outputs.
        """

    def update(
        self, full_batch: Batch, loss_fn: BatchLossFn, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Update the parameters of the module and all its submodules given the
        inputs and loss function. Gradients will automatically be calculated
        as necessary, and update_local() will be called on submodules.

        This should be called once on the root module, since it will amortize
        the cost of running backpropagation throughout the model.

        :param full_batch: a large batch of training examples.
        :param loss_fn: the global loss function.
        :param batch_size: if specified, split the batch up into minibatches
                           during the initial forward/backward passes.
        :return: a 1-D tensor of pre-update losses.
        """
        self._apply_cascade(lambda x: x._prepare_for_update())

        # Propagate all of the samples to accumulate inputs, outputs, and
        # gradients at all of the nodes.
        all_losses = []
        for indices, sub_batch in full_batch.batches(batch_size or len(full_batch)):
            losses = loss_fn(indices, self(sub_batch))
            all_losses.append(losses.detach())
            losses.sum().backward()

        self._apply_cascade(lambda x: x._updating())
        self._update(loss_fn)
        self._apply_cascade(lambda x: x._completed_update())

        return torch.cat(all_losses, dim=0)

    def _prepare_for_update(self):
        self._preparing_for_update = True

    def _updating(self):
        self._preparing_for_update = False

    def _completed_update(self):
        self._cached_inputs = []
        self._cached_outputs = []
        self._cached_grads = []

    def _apply_cascade(self, fn):
        def apply_fn(module):
            if isinstance(module, CascadeModule):
                fn(module)

        self.apply(apply_fn)

    def _update(
        self,
        loss_fn: BatchLossFn,
    ):
        inputs = Batch.cat(self._cached_inputs, dim=0)
        if self.requires_output_grad():
            outputs = Batch.cat(self._cached_outputs, dim=0)
            grads = Batch.cat(self._cached_grads, dim=0)
            assert len(inputs) == len(outputs), "invalid sample count fed through model"
            assert len(outputs) == len(
                grads
            ), "mismatching number of forward() and backward()"
        else:
            outputs, grads = None, None

        self.update_local(
            UpdateContext(
                inputs=inputs,
                loss_fn=loss_fn,
                outputs=outputs,
                output_grads=grads,
            )
        )


class CascadeSequential(CascadeModule):
    """
    Sequentially compose multiple cascade modules.
    """

    def __init__(self, sequence: Sequence[CascadeModule]):
        super().__init__()
        self.sequence = nn.ModuleList(sequence)

    def evaluate(self, inputs: Batch) -> Batch:
        out = inputs
        for layer in self.sequence:
            out = layer(out)
        return out

    def update_local(self, ctx: UpdateContext):
        for i in range(len(self.sequence))[::-1]:

            def child_loss_fn(indices: torch.Tensor, outputs: Batch) -> torch.Tensor:
                with torch.no_grad():
                    final_outputs = outputs
                    for layer in self.sequence[i + 1 :]:
                        final_outputs = layer(final_outputs)
                    return ctx.loss_fn(indices, final_outputs)

            child = self.sequence[i]
            child._update(child_loss_fn)


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

    def evaluate(self, inputs: Batch) -> Batch:
        return Batch.with_x(self.tree(inputs.x))

    def update_local(self, ctx: UpdateContext):
        h_tao = _CascadeTAO(
            xs=ctx.inputs.x,
            loss_fn=ctx.loss_fn,
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
                old_losses = torch.where(
                    cur_branch.decision(xs), right_losses, left_losses
                )
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

    def evaluate(self, inputs: Batch) -> Batch:
        return self.contained(inputs)

    def update_local(self, ctx: UpdateContext):
        self.contained.update_local(ctx)

    def update(
        self, full_batch: Batch, loss_fn: BatchLossFn, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        self.step.add_(1)
        if self.step.item() % self.interval == 0:
            result = super().update(full_batch, loss_fn, batch_size=batch_size)
            self.optimizer.zero_grad()
            return result
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
