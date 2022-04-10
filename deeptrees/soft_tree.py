from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .cascade import Batch, BatchLossFn, CascadeModule, UpdateContext
from .tree import ObliqueTreeBranch, TreeLeaf


class SoftTree(nn.Module):
    def __init__(
        self,
        weights: torch.Tensor,
        biases: torch.Tensor,
        masks: torch.Tensor,
        leaves: List[CascadeModule],
    ):
        super().__init__()
        self.weights = nn.Parameter(weights.detach().clone())
        self.biases = nn.Parameter(biases.detach().clone())
        self.register_buffer("masks", masks)
        self.leaves = nn.ModuleList(leaves)

    @classmethod
    def from_oblique(cls, tree: ObliqueTreeBranch) -> "SoftTree":
        weights = [tree.coef.view(1, -1)]
        biases = [-tree.threshold.view(1)]
        masks = []
        leaves = []

        prefix_mask = torch.zeros(
            (1, 2), dtype=tree.coef.dtype, device=tree.coef.device
        )
        for i, node in enumerate([tree.left, tree.right]):
            prefix_mask[:, :2] = 0
            prefix_mask[:, i] = 1.0

            if isinstance(node, TreeLeaf):
                masks.append(prefix_mask.clone())
                leaves.append(node)
                continue

            assert isinstance(
                node, ObliqueTreeBranch
            ), "only oblique branches are supported"
            sub_tree = cls.from_oblique(node)

            weights.append(sub_tree.weights)
            biases.append(sub_tree.biases)
            masks.append(
                torch.cat(
                    [prefix_mask.repeat(len(sub_tree.masks), 1), sub_tree.masks], dim=1
                )
            )
            leaves.extend(sub_tree.leaves)
            prefix_mask = torch.cat(
                [prefix_mask, torch.zeros_like(sub_tree.masks[:1])], dim=1
            )

        # When we created the first child's masks, we didn't know the length
        # of the second child's masks.
        padded_mask = torch.zeros(
            (sum(len(x) for x in masks), prefix_mask.shape[1]),
            device=prefix_mask.device,
            dtype=prefix_mask.dtype,
        )
        i = 0
        for x in masks:
            padded_mask[i : i + len(x), : x.shape[1]] = x
            i += len(x)

        return cls(
            weights=torch.cat(weights, dim=0),
            biases=torch.cat(biases, dim=0),
            masks=padded_mask,
            leaves=leaves,
        )

    def leaf_log_probs(self, xs: torch.Tensor) -> torch.Tensor:
        if not len(self.weights):
            return torch.ones(len(xs), dtype=xs.dtype, device=xs.device)
        logits = xs @ self.weights.t() + self.biases
        probs = torch.stack([F.logsigmoid(-logits), F.logsigmoid(logits)], dim=-1).view(
            logits.shape[0], logits.shape[1] * 2
        )
        return probs @ self.masks.t()

    def leaf_outputs(
        self, xs: torch.Tensor, leaf_indices: torch.Tensor
    ) -> torch.Tensor:
        if len(xs) == 0:
            return self.leaves[0](xs)

        index_sequence = torch.arange(len(leaf_indices), device=leaf_indices.device)
        inv_perm = torch.empty_like(leaf_indices)
        offset = 0
        outputs = []
        for i, leaf in enumerate(self.leaves):
            used = index_sequence[leaf_indices == i]
            if not len(used):
                continue
            leaf_out = leaf(xs[used])
            outputs.append(leaf_out)
            inv_perm[used] = torch.arange(offset, offset + len(leaf_out))
            offset += len(leaf_out)
        return torch.cat(outputs, dim=0)[inv_perm]


class CascadeSoftTree(CascadeModule):
    """
    Update a SoftTree with policy gradients (i.e. REINFORCE).
    """

    def __init__(self, tree: SoftTree, opt: optim.Optimizer, iters: int = 1):
        self.tree = tree
        self.opt = opt
        self.iters = iters

    def forward(self, batch: Batch, ctx: Optional[UpdateContext] = None) -> Batch:
        with torch.no_grad():
            log_probs = self.tree.leaf_log_probs(batch.x)
        leaf_indices = Categorical(probs=log_probs.exp()).sample()
        outs = batch.change_x(self.tree.leaf_outputs(batch.x, leaf_indices))
        if ctx is not None:
            ctx.cache_inputs(self, batch)
            ctx.cache_outputs(self, Batch.with_x(outs))
            ctx.cache_extra(self, Batch(leaf_indices=leaf_indices))
        return outs

    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        inputs = ctx.get_inputs(self)
        outputs = ctx.get_outputs(self)
        indices = ctx.get_extra(self)["leaf_indices"]
        losses = loss_fn(torch.arange(len(outputs.x)), outputs)
        losses = (losses - losses.mean()) / losses.std()
        for _ in range(self.iters):
            self.opt.zero_grad()
            log_probs = self.tree.leaf_log_probs(inputs.x.detach())
            (log_probs[range(len(indices)), indices] * losses).mean().backward()
            self.opt.step()
