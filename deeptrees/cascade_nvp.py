import itertools
import math
from abc import abstractmethod
from typing import Sequence, Tuple

import torch
from torch.distributions.normal import Normal

from .cascade import Batch, CascadeModule, UpdateContext


class CascadeNVPLayer(CascadeModule):
    """
    A layer in a RealNVP-style model.
    """

    def evaluate(self, x: Batch) -> Batch:
        out_vec, latents, log_det = self.evaluate_nvp(x.x)
        result = dict(x=out_vec)
        result.update({k: v for k, v in x.data.items() if k not in ["x", "log_det"]})
        next_latent_id = next(
            i for i in itertools.count() if f"latent_{i}" not in x.data
        )
        for i, latent in enumerate(latents):
            result[f"latent_{i+next_latent_id}"] = latent
        result["log_det"] = x.data.get("log_det", 0.0) + log_det
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


def apply_noise(b: Batch, noise_level=1.0 / 255.0) -> Batch:
    return Batch.with_x(b.x + torch.rand_like(b.x) * noise_level)


def nvp_loss(
    indices: torch.Tensor, batch: Batch, noise_level=1.0 / 255.0
) -> torch.Tensor:
    _ = indices  # this is self-supervised learning, so we have no targets
    log_det = batch.get("log_det", 0.0)
    latents = [batch.x]
    for i in itertools.count():
        k = f"latent_{i}"
        if k not in batch.data:
            break
        latents.append(batch.data[k].flatten(1))

    total_loss = log_det
    numel = 0
    for latent in latents:
        numel += latent[0].numel
        log_probs = Normal(0, 1).log_prob(latent)
        total_loss = total_loss + log_probs.flatten(1).sum(1)
    return ((total_loss / numel) + math.log(noise_level)) / math.log(2.0)
