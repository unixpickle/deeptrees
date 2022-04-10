import torch.nn as nn
import torch.optim as optim
from deeptrees.cascade import CascadeFlatten, CascadeFn
from deeptrees.cascade_init import (
    CascadeConvInit,
    CascadeFrozenInit,
    CascadeInit,
    CascadeParallelSumInit,
    CascadeRawInit,
    CascadeSequentialInit,
    CascadeTAOInit,
)
from deeptrees.fit_torch import TorchObliqueBranchBuilder
from deeptrees.soft_tree import CascadeSoftTreeInit


def conv_pool_soft_tree() -> CascadeInit:
    soft_tree_args = dict(
        tree_depth=2,
        iters=4,
        optimizer=lambda x: optim.Adam(x, lr=1e-4),
        verbose=True,
    )
    return CascadeSequentialInit(
        [
            CascadeFrozenInit(
                CascadeConvInit(
                    contained=CascadeSoftTreeInit(out_size=16, **soft_tree_args),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    loss="correlated",
                ),
                no_update=False,
            ),
            CascadeFrozenInit(
                CascadeConvInit(
                    contained=CascadeSoftTreeInit(out_size=32, **soft_tree_args),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    loss="correlated",
                ),
                no_update=False,
            ),
            CascadeRawInit(CascadeFn(nn.MaxPool2d(2))),
            CascadeRawInit(CascadeFlatten()),
            CascadeSoftTreeInit(
                out_size=128,
                **soft_tree_args,
            ),
            CascadeSoftTreeInit(
                out_size=10,
                **soft_tree_args,
            ),
        ]
    )


def conv_pool_tree_residual() -> CascadeInit:
    tao_args = dict(
        tree_depth=2,
        branch_builder=TorchObliqueBranchBuilder(
            max_epochs=1,
            optimizer_kwargs=dict(lr=1e-3, weight_decay=0.01),
        ),
        random_prob=0.1,
        reject_unimprovement=False,
        zero_init_out=True,
    )
    return CascadeSequentialInit(
        [
            CascadeParallelSumInit(
                [
                    CascadeRawInit(
                        lambda x: CascadeFn(
                            nn.Conv2d(1, 16, 3, padding=1, device=x.x.device)
                        )
                    ),
                    CascadeConvInit(
                        contained=CascadeTAOInit(out_size=16, **tao_args),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ]
            ),
            CascadeParallelSumInit(
                [
                    CascadeRawInit(
                        lambda x: CascadeFn(
                            nn.Conv2d(16, 32, 3, padding=1, device=x.x.device)
                        )
                    ),
                    CascadeConvInit(
                        contained=CascadeTAOInit(out_size=32, **tao_args),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ]
            ),
            CascadeRawInit(CascadeFn(nn.MaxPool2d(2))),
            CascadeRawInit(CascadeFlatten()),
            CascadeParallelSumInit(
                [
                    CascadeRawInit(
                        lambda x: CascadeFn(
                            nn.Linear(32 * 14 * 14, 128, device=x.x.device)
                        )
                    ),
                    CascadeTAOInit(out_size=128, **tao_args),
                ]
            ),
            CascadeParallelSumInit(
                [
                    CascadeRawInit(
                        lambda x: CascadeFn(nn.Linear(128, 10, device=x.x.device))
                    ),
                    CascadeTAOInit(out_size=10, **tao_args),
                ]
            ),
        ]
    )


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
