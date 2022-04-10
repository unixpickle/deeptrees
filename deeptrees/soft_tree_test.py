import torch

from .soft_tree import SoftTree
from .tree import ConstantTreeLeaf, ObliqueTreeBranch


def test_soft_tree():
    # Values taken by each dimension: ([0, 2], [-1, 1], [0, 1])
    tree = ObliqueTreeBranch(
        coef=torch.tensor([100.0, 0.0, 0.0]),
        threshold=torch.tensor([100.0]),
        left=ObliqueTreeBranch(
            coef=torch.tensor([0.0, 100.0, 0.0]),
            threshold=torch.tensor([0.0]),
            left=ConstantTreeLeaf(torch.tensor([1.0])),
            right=ObliqueTreeBranch(
                coef=torch.tensor([0.0, 0.0, 100.0]),
                threshold=torch.tensor([50.0]),
                left=ConstantTreeLeaf(torch.tensor([2.0])),
                right=ConstantTreeLeaf(torch.tensor([3.0])),
            ),
        ),
        right=ConstantTreeLeaf(torch.tensor([4.0])),
    )
    inputs = torch.tensor(
        [[x, y, z] for x in [0, 2] for y in [-1, 1] for z in [0, 1]],
        dtype=torch.float32,
    )
    expected = tree(inputs)

    soft_tree = SoftTree.from_oblique(tree)
    probs = soft_tree.leaf_log_probs(inputs).exp()
    indices = probs.argmax(-1)
    assert (
        probs[range(len(indices)), indices] > 0.99
    ).all(), "sigmoid should be saturated"
    actual = torch.cat(
        [soft_tree.leaves[i.item()](x[None]) for i, x in zip(indices, inputs)]
    )

    assert actual.shape == expected.shape
    assert (actual == expected).all().item()

    actual = soft_tree.leaf_outputs(inputs, indices)
    assert actual.shape == expected.shape
    assert (actual == expected).all().item()
