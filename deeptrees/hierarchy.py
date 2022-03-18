from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class UpdateContext:
    inputs: torch.Tensor

    # This function takes two arguments: (indices, outputs).
    # The indices argument specifies which inputs the outputs correspond to in
    # the full batch.
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    # Only specified if a module's "require-_output_grad" returns true.
    outputs: Optional[torch.Tensor] = None
    output_grads: Optional[torch.Tensor] = None


class HierarchyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._preparing_for_update = False
        self._cached_outputs = []
        self._cached_grads = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.evaluate(inputs)

        if not (self._preparing_for_update and self.requires_output_grad()):
            return out

        self._cached_outputs.append(out.detach().clone())

        class _CacheGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(_ctx, input):
                return input

            @staticmethod
            def backward(_ctx, grad_output):
                self._cached_grads.append(grad_output.detach().clone())
                return grad_output

        return _CacheGradFunc.apply(out)

    def requires_output_grad(self) -> bool:
        """
        If this method returns True, calls to update() will include the output
        and output gradient of the loss function for this module.
        """

    @abstractmethod
    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
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
        inputs: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        if self.requires_output_grad():
            outputs = torch.cat(self._cached_outputs, dim=0)
            grads = torch.cat(self._cached_grads, dim=0)
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
