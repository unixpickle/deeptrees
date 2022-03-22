import itertools
import math
from abc import abstractmethod
from typing import Sequence, Tuple

import torch
from torch.distributions.normal import Normal

from .cascade import Batch, CascadeModule, CascadeTAO, UpdateContext


class CascadeNVPLayer(CascadeModule):
    """
    A layer in a RealNVP-style model.
    """

    def evaluate(self, x: Batch) -> Batch:
        return self._results_batch(self.evaluate_nvp(x.x))

    def _results_batch(
        self,
        x: Batch,
        output: Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor],
    ) -> Batch:
        out_vec, latents, log_det = output
        result = dict(x=out_vec)
        result.update({k: v for k, v in x.items() if k not in ["x", "log_det"]})
        next_latent_id = next(i for i in itertools.count() if f"latent_{i}" not in x)
        for i, latent in enumerate(latents):
            result[f"latent_{i+next_latent_id}"] = latent
        result["log_det"] = x.get("log_det", 0.0) + log_det
        return result

    @abstractmethod
    def evaluate_nvp(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        """
        Evaluate the model on the input x.

        :return: a tuple (outputs, latents, log_det).
                 - outputs: the output latent of the layer to be sent downstream.
                 - latents: zero or more split off latent variables.
                 - log_det: the log-determinant of the Jacobian of this layer.
        """


class CascadeNVPPaddedLogit(CascadeNVPLayer):
    """
    An NVP layer that applies `logit(a + (1-2a)x)`.
    """

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def evaluate_nvp(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        y = self.alpha + (1 - 2 * self.alpha) * x
        logits = (y / (1 - y)).log()
        log_dets = (1 / y + 1 / (1 - y)).log() + ((1 - 2 * self.alpha)).log()
        return logits, [], log_dets.flatten(1).sum(1)

    def update_local(self, ctx: UpdateContext):
        _ = ctx


class CascadeNVPPartial(CascadeNVPLayer):
    """
    A layer that uses a subset of features to predict the scale and shift
    parameters for the remaining features. Wraps a sub-layer that does the
    actual scale/shift parameters.
    """

    def __init__(self, index_mask: torch.Tensor, sub_layer: CascadeModule):
        self.register_buffer("index_mask", index_mask)
        self.sub_layer = sub_layer

    def evaluate_nvp(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        predictions = self.sub_layer(x[:, self.index_mask])
        return self._output_for_predictions(x, predictions)

    def update_local(self, ctx: UpdateContext):
        def loss_fn(indices: torch.Tensor, outputs: Batch) -> torch.Tensor:
            sub_batch = ctx.inputs.at_indices(indices)
            out_tuple = self._output_for_predictions(sub_batch.x, outputs.x)
            out_batch = self._results_batch(sub_batch, out_tuple)
            return ctx.loss_fn(indices, out_batch)

        self.sub_layer._update(loss_fn)

    def _output_for_predictions(
        self, x: torch.Tensor, predictions: torch.Tensor
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        log_scale, bias = torch.split(predictions, predictions.shape[1] // 2, dim=1)
        output = torch.zeros_like(x)
        output[:, self.index_mask] = x[:, self.index_mask]
        output[:, ~self.index_mask] = x[:, ~self.index_mask] * log_scale.exp() + bias
        return output, [], log_scale.flatten(1).sum()


def quantization_noise(b: Batch, noise_level=1.0 / 255.0) -> Batch:
    return Batch.with_x(b.x + torch.rand_like(b.x) * noise_level)


def nvp_loss(
    indices: torch.Tensor, batch: Batch, noise_level=1.0 / 255.0
) -> torch.Tensor:
    _ = indices  # this is self-supervised learning, so we have no targets
    log_det = batch.get("log_det", 0.0)
    latents = [batch.x]
    for i in itertools.count():
        k = f"latent_{i}"
        if k not in batch:
            break
        latents.append(batch[k].flatten(1))

    total_loss = log_det
    numel = 0
    for latent in latents:
        numel += latent[0].numel
        log_probs = Normal(0, 1).log_prob(latent)
        total_loss = total_loss + log_probs.flatten(1).sum(1)
    return ((total_loss / numel) + math.log(noise_level)) / math.log(2.0)
