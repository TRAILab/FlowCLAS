"""Threshold-swept mean IoU (Jaccard) for pixel-level anomaly maps."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor

from .base import BaseMetric

_EPS = np.finfo(float).eps


class mIoU(BaseMetric):
    """Mean IoU over a grid of score thresholds (used by ``IoUThreshold``)."""

    full_state_update: bool = False

    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
        thresholds: Optional[int] = 100,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("thresholds", torch.linspace(0, 1, thresholds), persistent=False)
        self.add_state("num_batches", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("ious", default=torch.zeros(thresholds), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        preds = preds.reshape(preds.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        self.num_batches += 1
        for i, threshold in enumerate(self.thresholds):
            self.ious[i] += self._iou_at_threshold(preds, target, threshold)

    def _iou_at_threshold(self, pred: Tensor, target: Tensor, threshold: float) -> Tensor:
        target_mid = (torch.max(target) - torch.min(target)) / 2
        pred_pos = pred > threshold
        target_pos = target > target_mid
        intersection = torch.sum(pred_pos & target_pos, dim=1)
        union = torch.sum(pred_pos | target_pos, dim=1)
        return (intersection / (union + _EPS)).mean()

    def compute(self, reduce: bool = True) -> Tensor:
        ious = self.ious / self.num_batches
        return ious.mean() if reduce else ious
