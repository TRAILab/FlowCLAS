"""Custom anomaly-detection metrics and metric collection helpers."""

import importlib
import logging
from typing import Any

import torchmetrics
from omegaconf import DictConfig, ListConfig

from anomalib.metrics import (
    _validate_metrics_dict as validate_metrics_dict,
    metric_collection_from_dicts,
)
from anomalib.metrics.collection import AnomalibMetricCollection

from .analysis import ECEMetric, MCEMetric
from .auroc import BinaryAUROC
from .average_precision import BinaryAP
from .base import BaseMetric
from .fpr import FPR95
from .recall import RecallAt01, RecallAt05, RecallAtP
from .threshold import F1Threshold, IoUThreshold

__all__ = [
    "BaseMetric",
    "BinaryAUROC",
    "BinaryAP",
    "ECEMetric",
    "F1Threshold",
    "FPR95",
    "IoUThreshold",
    "MCEMetric",
    "RecallAt01",
    "RecallAt05",
    "RecallAtP",
    "create_metric_collection",
    "metric_collection_from_names",
]

logger = logging.getLogger(__name__)


def metric_collection_from_names(
    metric_names: list[str], prefix: str | None
) -> AnomalibMetricCollection:
    """Build a metric collection from registered class names.

    Resolution order: ``core.utils.metrics``, then ``anomalib.metrics``, then ``torchmetrics``.
    """
    metrics_module = importlib.import_module("core.utils.metrics")
    anomalib_metrics_module = importlib.import_module("anomalib.metrics")
    metrics = AnomalibMetricCollection([], prefix=prefix)
    for metric_name in metric_names:
        if hasattr(metrics_module, metric_name):
            metric_cls = getattr(metrics_module, metric_name)
            metrics.add_metrics(metric_cls())
        elif hasattr(anomalib_metrics_module, metric_name):
            try:
                metric_cls = getattr(anomalib_metrics_module, metric_name)
                metrics.add_metrics(metric_cls())
            except TypeError:
                logger.warning(
                    "Incorrect constructor arguments for %s from anomalib.metrics.",
                    metric_name,
                )
        elif hasattr(torchmetrics, metric_name):
            try:
                metric_cls = getattr(torchmetrics, metric_name)
                metrics.add_metrics(metric_cls())
            except TypeError:
                logger.warning(
                    "Incorrect constructor arguments for %s from torchmetrics.",
                    metric_name,
                )
        else:
            logger.warning(
                "No metric named %s in core.utils.metrics, anomalib.metrics, or torchmetrics.",
                metric_name,
            )
    return metrics


def create_metric_collection(
    metrics: list[BaseMetric] | list[str] | dict[str, dict[str, Any]],
    prefix: str | None,
) -> AnomalibMetricCollection:
    """Create a metric collection from metric instances, names, or Lightning-style dict specs."""
    if isinstance(metrics, list) and all(isinstance(metric, BaseMetric) for metric in metrics):
        return AnomalibMetricCollection(metrics, prefix=prefix)

    if isinstance(metrics, DictConfig | dict):
        validate_metrics_dict(metrics)
        return metric_collection_from_dicts(metrics, prefix)

    if isinstance(metrics, ListConfig | list):
        if not all(isinstance(metric, str) for metric in metrics):
            raise ValueError(f"Metric name list must contain only strings, got {metrics}")
        return metric_collection_from_names(metrics, prefix)

    raise ValueError(f"metrics must be a list or dict, found {type(metrics)}")
