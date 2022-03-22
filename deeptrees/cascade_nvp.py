import itertools
import math
from abc import abstractmethod
from typing import List, Sequence, Tuple

import torch
from torch.distributions.normal import Normal

from .cascade import Batch, CascadeModule, CascadeSequential, UpdateContext


class CascadeNVPLayer(CascadeModule):
    """
    A layer in a RealNVP-style model.
    """

    def evaluate(self, x: Batch) -> Batch:
        return self._results_batch(x, self.evaluate_nvp(x.x))

    def _results_batch(
        self,
        x: Batch,
        output: Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor],
    ) -> Batch:
        out_vec, latents, log_det = output
        result = Batch.with_x(out_vec)
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

        :param x: the input tensor from the previous layer.
        :return: a tuple (outputs, latents, log_det).
                 - outputs: the output latent of the layer to be sent downstream.
                 - latents: zero or more split off latent variables.
                 - log_det: the log-determinant of the Jacobian of this layer.
        """

    @abstractmethod
    def invert(self, outputs: torch.Tensor, latents: Sequence[torch.Tensor]):
        """
        Compute the input x given the outputs and output latents.
        """

    @property
    def num_latents(self) -> int:
        """
        Get the number of latents returned by evaluate_nvp().
        """
        return 0


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

    def invert(self, outputs: torch.Tensor, latents: Sequence[torch.Tensor]):
        assert not len(latents)
        return (outputs.sigmoid() - self.alpha) / (1 - 2 * self.alpha)

    def update_local(self, ctx: UpdateContext):
        _ = ctx


class CascadeNVPPartial(CascadeNVPLayer):
    """
    A layer that uses a subset of features to predict the scale and shift
    parameters for the remaining features. Wraps a sub-layer that computes the
    actual scale/shift parameters.
    """

    def __init__(self, feature_mask: torch.Tensor, sub_layer: CascadeModule):
        super().__init__()
        self.register_buffer("feature_mask", feature_mask)
        self.sub_layer = sub_layer

    def evaluate_nvp(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        predictions = self.sub_layer(
            Batch.with_x(x[:, self.feature_mask]), cache_inputs=False
        )
        return self._output_for_predictions(x, predictions.x)

    def invert(self, outputs: torch.Tensor, latents: Sequence[torch.Tensor]):
        assert not len(latents)
        predictions = self.sub_layer(
            Batch.with_x(outputs[:, self.feature_mask]), cache_inputs=False
        )
        return self._output_for_predictions(outputs, predictions.x, inverse=True)[0]

    def update_local(self, ctx: UpdateContext):
        def loss_fn(indices: torch.Tensor, outputs: Batch) -> torch.Tensor:
            sub_batch = ctx.inputs.at_indices(indices)
            out_tuple = self._output_for_predictions(sub_batch.x, outputs.x)
            out_batch = self._results_batch(sub_batch, out_tuple)
            return ctx.loss_fn(indices, out_batch)

        self.sub_layer._update(
            loss_fn, inputs=Batch.with_x(ctx.inputs.x[:, self.feature_mask])
        )

    def _output_for_predictions(
        self, x: torch.Tensor, predictions: torch.Tensor, inverse: bool = False
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        log_scale, bias = torch.split(predictions, predictions.shape[1] // 2, dim=1)
        output = torch.zeros_like(x)
        output[:, self.feature_mask] = x[:, self.feature_mask]
        if inverse:
            output[:, ~self.feature_mask] = (x[:, ~self.feature_mask] - bias) * (
                -log_scale
            ).exp()
        else:
            output[:, ~self.feature_mask] = (
                x[:, ~self.feature_mask] * log_scale.exp() + bias
            )
        return output, [], log_scale.flatten(1).sum(1)


class CascadeNVPSequential(CascadeNVPLayer, CascadeSequential):
    def evaluate_nvp(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]:
        out = Batch.with_x(x)
        for layer in self.sequence:
            out = layer(out)
        return out.x, latents_from_batch(out), out["log_det"]

    def invert(self, outputs: torch.Tensor, latents: Sequence[torch.Tensor]):
        latent_stack = list(latents)
        for layer in self.sequence[::-1]:
            sub_latents = []
            if layer.num_latents:
                assert len(latent_stack) >= layer.num_latents
                sub_latents = latent_stack[-layer.num_latents :]
                del latent_stack[-layer.num_latents :]
            outputs = layer.invert(outputs, sub_latents)
        assert len(latent_stack) == 0
        return outputs

    @property
    def num_latents(self) -> int:
        return sum(x.num_latents for x in self.sequence)


def quantization_noise(b: torch.Tensor, noise_level=1.0 / 255.0) -> torch.Tensor:
    return b + torch.rand_like(b) * noise_level


def nvp_loss(
    indices: torch.Tensor, batch: Batch, noise_level=1.0 / 255.0
) -> torch.Tensor:
    _ = indices  # this is self-supervised learning, so we have no targets
    log_det = batch.get("log_det", 0.0)
    latents = [batch.x.flatten(1)] + [x.flatten(1) for x in latents_from_batch(batch)]

    total_loss = log_det
    numel = 0
    for latent in latents:
        numel += latent.shape[1]
        log_probs = Normal(0, 1).log_prob(latent)
        total_loss = total_loss + log_probs.sum(1)
    return -((total_loss / numel) + math.log(noise_level)) / math.log(2.0)


def latents_from_batch(batch: Batch) -> List[torch.Tensor]:
    result = []
    for i in itertools.count():
        k = f"latent_{i}"
        if k not in batch:
            break
        result.append(batch[k])
    return result
