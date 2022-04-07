import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch.nn as nn

from .tree import Tree, TreeLeaf


@dataclass
class TreeLeafUsage:
    counts: Dict[TreeLeaf, int]

    def num_used(self) -> int:
        return sum(int(x > 0) for x in self.counts.values())

    def entropy(self) -> float:
        total_usage = sum(self.counts.values())
        result = 0.0
        for x in self.counts.values():
            p = x / total_usage
            result -= p * math.log2(p)
        return result


@contextmanager
def track_tree_usage(module: nn.Module) -> Iterator[Dict[str, TreeLeafUsage]]:
    trees = dict(named_trees(module))
    handles = []
    results = {}
    for tree_name, tree in trees.items():
        results[tree_name] = TreeLeafUsage(dict())

        def forward_hook(self, input, output, out_dict=results[tree_name].counts):
            _ = output
            out_dict[self] += len(input[0])

        for leaf in tree.iterate_leaves():
            results[tree_name].counts[leaf] = 0
            handles.append(leaf.register_forward_hook(forward_hook))

    yield results

    for hook in handles:
        hook.remove()


def named_trees(module: nn.Module) -> Iterator[Tuple[str, Tree]]:
    """
    Iterate over all of the (root) trees contained in the module.
    """
    if isinstance(module, Tree):
        yield "", module
    else:
        for name, child in module.named_children():
            for sub_name, tree in named_trees(child):
                yield (f"{name}.{sub_name}" if sub_name else name), tree
