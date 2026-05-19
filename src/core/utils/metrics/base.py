"""Base class for evaluation metrics."""

from abc import ABC

import torch
from torchmetrics import Metric


class BaseMetric(Metric, ABC):
    """Abstract metric; subclasses must implement ``update`` and ``compute``."""

    def compute(self) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement compute")

    def update(self, *args, **kwargs) -> None:  # noqa: ARG002
        raise NotImplementedError("Subclass must implement update")
