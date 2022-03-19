from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

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
    An "HModule" is short for hierarchy module.
    """

    def __init__(self):
        super().__init__()
        self._preparing_for_update = False
        self._cached_outputs = []
        self._cached_grads = []

    def forward(self, inputs: Batch) -> Batch:
        out = self.evaluate(inputs)

        if not (self._preparing_for_update and self.requires_output_grad()):
            return out

        self._cached_outputs.append(out.detach().clone())

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
    def update(self, ctx: UpdateContext):
        """
        Update the parameters of the module given the inputs, loss function,
        and possibly gradients of the loss function w.r.t. the outputs.
        """

    def _prepare_for_update(self):
        self._preparing_for_update = True
        self._cached_outputs = []
        self._cached_grads = []

    def _updating(self):
        self._preparing_for_update = False

    def _update(
        self,
        inputs: Batch,
        loss_fn: BatchLossFn,
    ):
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
