"""Expected and maximum calibration error for binary anomaly scores."""

from typing import Any

import torch
from torch import Tensor
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification.calibration_error import _binning_bucketize

from ..base import BaseMetric


class CalibrationMetric(BinaryCalibrationError, BaseMetric):
    """Binary calibration error with optional monotonicity check on binned accuracies."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        n_bins: int = 15,
        norm: str = "max",
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_bins=n_bins, norm=norm, ignore_index=ignore_index, **kwargs)

    def compute(self) -> Tensor:
        return super().compute()

    @torch.no_grad()
    def check_monotonicity(self) -> bool:
        """True if per-bin accuracies are non-decreasing across confidence bins."""
        bin_boundaries = torch.linspace(
            0, 1, self.n_bins + 1, dtype=self.confidences.dtype, device=self.confidences.device
        )
        acc_bin, _, _ = _binning_bucketize(self.confidences, self.accuracies, bin_boundaries)
        return torch.all(torch.diff(acc_bin) >= 0)


class ECEMetric(CalibrationMetric):
    """Expected calibration error (L1 norm over bins)."""

    def __init__(self, norm: str = "l1", **kwargs: Any) -> None:
        super().__init__(norm=norm, **kwargs)


class MCEMetric(CalibrationMetric):
    """Maximum calibration error (L-inf norm over bins)."""

    def __init__(self, norm: str = "max", **kwargs: Any) -> None:
        super().__init__(norm=norm, **kwargs)
