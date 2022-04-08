import torch.nn as nn
from deeptrees.cascade import CascadeFlatten, CascadeFn
from deeptrees.cascade_init import (
    CascadeConvInit,
    CascadeFrozen,
    CascadeInit,
    CascadeParallelSumInit,
    CascadeRawInit,
    CascadeSequentialInit,
    CascadeTAOInit,
)
from deeptrees.fit_torch import TorchObliqueBranchBuilder


def conv_pool_tree_residual() -> CascadeInit:
    tao_args = dict(
        tree_depth=2,
        branch_builder=TorchObliqueBranchBuilder(
            max_epochs=10,
            optimizer_kwargs=dict(lr=1e-3, weight_decay=0.01),
        ),
        random_prob=0.1,
        reject_unimprovement=False,
        # zero_init_out=True,
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
            max_epochs=10,
            optimizer_kwargs=dict(lr=1e-3, weight_decay=0.01),
        ),
        random_prob=0.0,
        reject_unimprovement=False,
        # replicate_leaves=True,
    )
    return CascadeSequentialInit(
        [
            CascadeFrozen(
                CascadeConvInit(
                    contained=CascadeTAOInit(out_size=16, **tao_args),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
            CascadeFrozen(
                CascadeConvInit(
                    contained=CascadeTAOInit(out_size=32, **tao_args),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
            CascadeRawInit(CascadeFn(nn.MaxPool2d(2))),
            CascadeRawInit(CascadeFlatten()),
            CascadeFrozen(
                CascadeTAOInit(out_size=128, **tao_args),
            ),
            CascadeFrozen(
                CascadeTAOInit(out_size=10, **tao_args),
                no_update=False,
            ),
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
