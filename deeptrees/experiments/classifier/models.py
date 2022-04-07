import torch.nn as nn
from deeptrees.cascade import CascadeFlatten, CascadeFn
from deeptrees.cascade_init import (
    CascadeConvInit,
    CascadeInit,
    CascadeRawInit,
    CascadeSequentialInit,
    CascadeTAOInit,
)
from deeptrees.fit_torch import TorchObliqueBranchBuilder


def conv_pool_tree() -> CascadeInit:
    tao_args = dict(
        tree_depth=2,
        branch_builder=TorchObliqueBranchBuilder(
            max_epochs=1,
            optimizer_kwargs=dict(lr=1e-3, weight_decay=0.01),
        ),
        random_prob=0.1,
        reject_unimprovement=False,
        replicate_leaves=True,
    )
    return CascadeSequentialInit(
        [
            CascadeConvInit(
                contained=CascadeTAOInit(out_size=16, **tao_args),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            CascadeConvInit(
                contained=CascadeTAOInit(out_size=32, **tao_args),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            CascadeRawInit(CascadeFn(nn.MaxPool2d(2))),
            CascadeRawInit(CascadeFlatten()),
            CascadeTAOInit(out_size=128, **tao_args),
            CascadeTAOInit(out_size=10, **tao_args),
        ]
    )


def conv_pool_baseline_no_acts() -> CascadeInit:
    return CascadeSequentialInit(
        [
            CascadeRawInit(
                lambda x: CascadeFn(nn.Conv2d(1, 16, 3, padding=1, device=x.x.device))
            ),
            CascadeRawInit(
                lambda x: CascadeFn(nn.Conv2d(16, 32, 3, padding=1, device=x.x.device))
            ),
            CascadeRawInit(CascadeFn(nn.MaxPool2d(2))),
            CascadeRawInit(CascadeFlatten()),
            CascadeRawInit(
                lambda x: CascadeFn(nn.Linear(32 * 14 * 14, 128, device=x.x.device))
            ),
            CascadeRawInit(lambda x: CascadeFn(nn.Linear(128, 10, device=x.x.device))),
        ]
    )


def conv_pool_baseline_relu() -> CascadeInit:
    return CascadeSequentialInit(
        [
            CascadeRawInit(
                lambda x: CascadeFn(nn.Conv2d(1, 16, 3, padding=1, device=x.x.device))
            ),
            CascadeRawInit(CascadeFn(nn.ReLU())),
            CascadeRawInit(
                lambda x: CascadeFn(nn.Conv2d(16, 32, 3, padding=1, device=x.x.device))
            ),
            CascadeRawInit(CascadeFn(nn.ReLU())),
            CascadeRawInit(CascadeFn(nn.MaxPool2d(2))),
            CascadeRawInit(CascadeFlatten()),
            CascadeRawInit(
                lambda x: CascadeFn(nn.Linear(32 * 14 * 14, 128, device=x.x.device))
            ),
            CascadeRawInit(CascadeFn(nn.ReLU())),
            CascadeRawInit(lambda x: CascadeFn(nn.Linear(128, 10, device=x.x.device))),
        ]
    )