"""Binary AUROC with optional ignore_index filtering."""

from __future__ import annotations

from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics.classification.roc import BinaryROC
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
)
from torchmetrics.utilities.compute import auc

from anomalib.metrics.plotting_utils import plot_figure

from .base import BaseMetric


class BinaryAUROC(BinaryROC, BaseMetric):
    """Area under the ROC curve for binary scores."""

    def __init__(self, thresholds: int = 100, **kwargs) -> None:
        super().__init__(thresholds=thresholds, validate_args=False, **kwargs)

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

    def compute(self) -> Tensor:
        fpr, tpr, _ = super().compute()
        return auc(fpr, tpr, reorder=True)

    def generate_figure(self) -> tuple[Figure, str]:
        """ROC curve with diagonal baseline for logging."""
        fpr, tpr, _ = super().compute()
        auroc = self.compute()
        fig, axis = plot_figure(
            fpr, tpr, auroc, (0.0, 1.0), (0.0, 1.0), "False Positive Rate", "True Positive Rate", "lower right", "ROC"
        )
        axis.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", figure=fig)
        return fig, "ROC"
