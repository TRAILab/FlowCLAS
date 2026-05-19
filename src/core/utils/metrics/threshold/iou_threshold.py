"""Validation threshold that maximizes mean IoU over swept score thresholds."""

import logging

import torch

from anomalib.metrics.threshold.base import BaseThreshold

from core.utils.metrics.miou import mIoU


class IoUThreshold(mIoU, BaseThreshold):
    """Adaptive threshold: score at maximum mIoU over swept thresholds."""

    def __init__(self, default_value: float = 0.5, thresholds: int = 100, **kwargs) -> None:
        super().__init__(thresholds=thresholds, validate_args=False, **kwargs)
        self.add_state("value", default=torch.tensor(default_value), dist_reduce_fx="mean", persistent=True)
        self.seen_anomaly: bool = False

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.seen_anomaly = self.seen_anomaly or any(1 in batch for batch in target)
        super().update(preds, target)

    def compute(self) -> torch.Tensor:
        if not self.seen_anomaly:
            logging.warning(
                "Validation set has no anomalous samples; adaptive threshold may be unreliable."
            )
        ious = super().compute(reduce=False)
        if self.thresholds.dim() == 0:
            self.value = self.thresholds
        else:
            self.value = self.thresholds[torch.argmax(ious)]
        return self.value

    def __repr__(self) -> str:
        return f"{super().__repr__()} (value={self.value:.2f})"
