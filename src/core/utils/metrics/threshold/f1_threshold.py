"""Validation threshold that maximizes F1 on the PR curve."""

import logging
from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
)

from anomalib.metrics.threshold.base import BaseThreshold


class F1Threshold(BinaryPrecisionRecallCurve, BaseThreshold):
    """Adaptive threshold: score at maximum F1 over swept thresholds."""

    def __init__(self, default_value: float = 0.5, thresholds: int = 100, **kwargs) -> None:
        super().__init__(thresholds=thresholds, validate_args=False, **kwargs)
        self.add_state("value", default=torch.tensor(default_value), dist_reduce_fx="mean", persistent=True)
        self.seen_anomaly: bool = False

    def _update(self, preds: Tensor, target: Tensor) -> None:
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

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.seen_anomaly = self.seen_anomaly or any(1 in batch for batch in target)
        self._update(preds, target)

    def compute(self) -> torch.Tensor:
        if not self.seen_anomaly:
            logging.warning(
                "Validation set has no anomalous samples; adaptive threshold may be unreliable."
            )
        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> None:
        super().load_state_dict(state_dict)
        self.value = state_dict["value"]

    def __repr__(self) -> str:
        return f"{super().__repr__()} (value={self.value:.2f})"
