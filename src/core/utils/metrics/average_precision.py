"""Binary average precision (area under the PR curve)."""

from __future__ import annotations

from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
)
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.metrics.plotting_utils import plot_figure

from .base import BaseMetric


class BinaryAP(BinaryPrecisionRecallCurve, BaseMetric):
    """Average precision for binary anomaly scores."""

    def __init__(self, thresholds: int = 100, **kwargs) -> None:
        super().__init__(thresholds=thresholds, validate_args=False, **kwargs)
        self.num_pos_targets = 0
        self.num_targets = 0

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

    def update(self, preds: Tensor, target: Tensor) -> None:
        self._update(preds, target)
        self.num_pos_targets += (dim_zero_cat(target) == 1).sum()
        self.num_targets += dim_zero_cat(target).numel()

    def compute(self) -> Tensor:
        prec, rec, _ = super().compute()
        return auc(rec, prec, reorder=True)

    def generate_figure(self) -> tuple[Figure, str]:
        """PR curve with prevalence baseline for logging."""
        prec, rec, _ = super().compute()
        ap = self.compute()
        invalid = (prec == 0) & (rec == 0)
        prec, rec = prec[~invalid], rec[~invalid]
        fig, axis = plot_figure(rec, prec, ap, (0.0, 1.0), (0.0, 1.0), "Recall", "Precision", "best", "AP")
        rate = self.num_pos_targets / self.num_targets
        axis.plot(
            (0, 1),
            (rate.detach().cpu(), rate.detach().cpu()),
            color="navy",
            lw=2,
            linestyle="--",
            figure=fig,
        )
        return fig, "AP"
