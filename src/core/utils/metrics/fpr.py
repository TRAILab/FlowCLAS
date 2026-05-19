"""False positive rate at a fixed true positive rate."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics.classification.roc import BinaryROC
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
)

from .base import BaseMetric


class FPRAtTPR(BinaryROC, BaseMetric):
    """Minimum FPR among points where TPR >= ``tpr``."""

    def __init__(self, tpr: float, thresholds: int = 100, **kwargs) -> None:
        super().__init__(thresholds=thresholds, validate_args=False, **kwargs)
        self.tpr = tpr

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
        fpr, tpr, _ = super().compute()
        possible_fprs = fpr[tpr >= self.tpr]
        if possible_fprs.numel() == 0:
            return torch.tensor(1.0, device=self.device)
        return possible_fprs.min()


class FPR95(FPRAtTPR):
    """FPR at TPR = 0.95 (common anomaly-detection benchmark)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(tpr=0.95, **kwargs)
