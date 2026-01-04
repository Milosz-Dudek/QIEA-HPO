"""Exports for benchmark task factories."""

from .tabular_tasks import (
    get_tabular_dataset,
    make_mlp_objective,
    make_svm_objective,
    make_xgboost_objective,
)
from .vision_tasks import (
    make_cifar10_objective,
    make_mnist_objective,
)

__all__ = [
    "get_tabular_dataset",
    "make_mlp_objective",
    "make_svm_objective",
    "make_xgboost_objective",
    "make_cifar10_objective",
    "make_mnist_objective",
]
