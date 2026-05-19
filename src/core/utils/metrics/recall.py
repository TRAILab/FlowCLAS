"""Recall at a fixed minimum precision."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics.classification import BinaryRecallAtFixedPrecision
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
)

from .base import BaseMetric


class RecallAtP(BinaryRecallAtFixedPrecision, BaseMetric):
    """Recall when precision is at least ``precision``."""

    def __init__(self, precision: float, thresholds: int = 100, **kwargs) -> None:
        super().__init__(min_precision=precision, thresholds=thresholds, validate_args=False, **kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            _binary_precision_recall_curve_tensor_validation(preds, target, self.ignore_index)
        mask = target != self.ignore_index
        preds_, target_ = preds[mask], target[mask]
        state = _binary_precision_recall_curve_update(preds_, target_, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> torch.Tensor:
        recall, _ = super().compute()
        return recall


class RecallAt05(RecallAtP):
    """Recall at precision >= 0.5."""

    def __init__(self, thresholds: int = 100, **kwargs) -> None:
        super().__init__(precision=0.5, thresholds=thresholds, **kwargs)


class RecallAt01(RecallAtP):
    """Recall at precision >= 0.1."""

    def __init__(self, thresholds: int = 100, **kwargs) -> None:
        super().__init__(precision=0.1, thresholds=thresholds, **kwargs)
