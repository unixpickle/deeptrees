from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .cascade import Batch, BatchLossFn, CascadeModule, UpdateContext
from .cascade_init import CascadeInit, random_tree, replicate_leaves
from .tree import ObliqueTreeBranch, Tree, TreeBranch, TreeLeaf


class SoftTree(Tree):
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
        result = None
        for i, leaf in enumerate(self.leaves):
            used = index_sequence[leaf_indices == i]
            if not len(used):
                continue
            leaf_out = leaf(xs[used])
            if result is None:
                result = torch.empty(
                    (len(xs), leaf_out.shape[1]),
                    dtype=leaf_out.dtype,
                    device=leaf_out.device,
                )
            result[used] = leaf_out
        return result

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            log_probs = self.leaf_log_probs(xs)
        leaf_indices = Categorical(probs=log_probs.exp()).sample()
        return self.leaf_outputs(xs, leaf_indices)

    def forward_chunks(
        self, indices: torch.Tensor, xs: torch.Tensor
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        yield indices, self(xs)

    def prune(self, xs: torch.Tensor) -> Tree:
        _ = xs
        return self

    def map_branches(self, fn: Callable[[TreeBranch], TreeBranch]) -> Tree:
        _ = fn
        return self

    def map_leaves(self, fn: Callable[[TreeLeaf], TreeLeaf]) -> Tree:
        return self.__class__(
            weights=self.weights,
            biases=self.biases,
            masks=self.masks,
            leaves=[fn(x) for x in self.leaves],
        )

    def iterate_leaves(self) -> Iterator[TreeLeaf]:
        yield from self.leaves


class CascadeSoftTree(CascadeModule):
    """
    Update a SoftTree with policy gradients (i.e. REINFORCE).
    """

    def __init__(
        self,
        tree: SoftTree,
        opt: optim.Optimizer,
        iters: int = 1,
        entropy_coef: float = 0.01,
        epsilon: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__()
        self.tree = tree
        self.opt = opt
        self.iters = iters
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.verbose = verbose

    def forward(self, batch: Batch, ctx: Optional[UpdateContext] = None) -> Batch:
        if ctx is None:
            # Call forward() on self.tree directly so that tree-based
            # usage analysis can use forward hooks on Tree sub-classes.
            return batch.change_x(self.tree(batch.x))

        with torch.no_grad():
            log_probs = self.tree.leaf_log_probs(batch.x)
        leaf_indices = Categorical(probs=log_probs.exp()).sample()
        outs = batch.change_x(self.tree.leaf_outputs(batch.x, leaf_indices))
        ctx.cache_inputs(self, batch)
        ctx.cache_extra(self, Batch(leaf_indices=leaf_indices))
        return outs

    def update_local(self, ctx: UpdateContext, loss_fn: BatchLossFn):
        _ = loss_fn

        inputs = ctx.get_inputs(self)
        indices = ctx.get_extra(self)["leaf_indices"]
        indices_range = torch.arange(len(indices), device=indices.device)

        with torch.no_grad():
            old_probs = self.tree.leaf_log_probs(inputs.x)[indices_range, indices]

        losses = ctx.get_losses()
        advs = -(losses - losses.mean()) / losses.std()
        if self.verbose:
            print(
                f"  - performing {self.iters} iterations of PPO (in_shape={inputs.x.shape})"
            )
        for i in range(self.iters):
            self.opt.zero_grad()
            log_probs = self.tree.leaf_log_probs(inputs.x.detach())
            prob_ratio = (log_probs[indices_range, indices] - old_probs).exp()
            prob_ratio_clip = prob_ratio.clamp(1 - self.epsilon, 1 + self.epsilon)
            ppo_loss = torch.minimum(prob_ratio * advs, prob_ratio_clip * advs).mean()
            entropy = Categorical(logits=log_probs).entropy().mean()
            loss = -(ppo_loss + self.entropy_coef * entropy)
            loss.backward()
            self.opt.step()
            if self.verbose:
                clip_frac = (prob_ratio != prob_ratio_clip).float().mean().item()
                print(
                    f"   - step {i}: adv={ppo_loss.item()} entropy={entropy.item()} clip_frac={clip_frac}"
                )


@dataclass
class CascadeSoftTreeInit(CascadeInit):
    """
    Randomly initialize a CascadeSoftTree with random splits of the data.
    """

    out_size: int
    tree_depth: int
    iters: int = 1
    entropy_coef: float = 0.01
    epsilon: int = 0.1
    zero_init_out: bool = False
    replicate_leaves: bool = False
    verbose: bool = False
    optimizer: Callable[
        [Iterable[nn.Parameter]], optim.Optimizer
    ] = lambda x: optim.Adam(x, lr=1e-3)

    def __call__(
        self, inputs: Batch, targets: Optional[Batch] = None
    ) -> Tuple[CascadeModule, Batch]:
        _ = targets
        tree = random_tree(
            inputs.x,
            self.out_size,
            self.tree_depth,
            zero_init_out=self.zero_init_out,
        )
        if self.replicate_leaves:
            replicate_leaves(tree)
        soft_tree = SoftTree.from_oblique(tree)
        module = CascadeSoftTree(
            soft_tree,
            opt=self.optimizer(soft_tree.parameters()),
            iters=self.iters,
            entropy_coef=self.entropy_coef,
            epsilon=self.epsilon,
            verbose=self.verbose,
        )
        with torch.no_grad():
            inputs = module(inputs)
        return module, inputs
