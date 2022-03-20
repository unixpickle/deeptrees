from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn


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
            for k, v in element.values.items():
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
        return len(next(self.data.values()))


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


class HModule(nn.Module):
    """
    An "HModule" is short for "hierarchy module". These are modules that can be
    used in a deep network of decision trees. Unlike regular deep learning
    modules, each module gets a loss function that determines, per input, what
    the loss would be regardless of the output for that input.
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
    ):
        """
        Update the parameters of the module and all its submodules given the
        inputs and loss function. Gradients will automatically be calculated
        as necessary, and update_local() will be called on submodules.

        This should be called once on the root module, since it will amortize
        the cost of running backpropagation throughout the model.
        """
        self._apply_hmodules(lambda x: x._preparing_for_update())

        # Propagate all of the samples to accumulate inputs, outputs, and
        # gradients at all of the nodes.
        for indices, sub_batch in full_batch.batches(batch_size or len(full_batch)):
            loss = loss_fn(indices, self(sub_batch)).sum()
            loss.backward()

        self._apply_hmodules(lambda x: x._updating())
        self._update(loss_fn)
        self._apply_hmodules(lambda x: x._completed_update())

    def _prepare_for_update(self):
        self._preparing_for_update = True

    def _updating(self):
        self._preparing_for_update = False

    def _completed_update(self):
        self._cached_inputs = []
        self._cached_outputs = []
        self._cached_grads = []

    def _apply_hmodules(self, fn):
        def apply_fn(module):
            if isinstance(module, HModule):
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

        self.update(
            UpdateContext(
                inputs=inputs,
                loss_fn=loss_fn,
                outputs=outputs,
                output_grads=grads,
            )
        )


class HSequential(HModule):
    def __init__(self, sequence: Sequence[HModule]):
        self.sequence = nn.ModuleList(sequence)

    def evaluate(self, inputs: Batch) -> Batch:
        out = inputs
        for layer in self.sequence:
            out = layer(inputs)
        return out

    def update_local(self, ctx: UpdateContext):
        for i in range(len(self.sequence))[::-1]:

            def child_loss_fn(indices: torch.Tensor, outputs: Batch) -> torch.Tensor:
                final_outputs = outputs
                for layer in self.sequence[i + 1 :]:
                    final_outputs = layer(final_outputs)
                return ctx.loss_fn(indices, final_outputs)

            child = self.sequence[i]
            child._update(child_loss_fn)
