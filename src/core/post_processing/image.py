"""Batch image visualizer for predict-time exports (heatmap, mask, histogram)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from anomalib.data.utils import read_image
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib.utils.visualization import BaseVisualizer, GeneratorResult, VisualizationStep

from core.utils import TaskType


class ImageResult:
    """Per-image tensors and rendered panels used by :class:`ImageVisualizer`."""

    def __init__(
        self,
        image: np.ndarray,
        pred_score: float | None = None,
        pred_label: str | int | None = None,
        anomaly_map: np.ndarray | None = None,
        gt_mask: np.ndarray | None = None,
        pred_mask: np.ndarray | None = None,
        gt_boxes: np.ndarray | None = None,
        pred_boxes: np.ndarray | None = None,
        box_labels: np.ndarray | None = None,
    ) -> None:
        self.image = image
        self.pred_score = pred_score
        self.pred_label = pred_label
        self.anomaly_map = anomaly_map
        self.gt_mask = gt_mask
        self.pred_mask = pred_mask
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes
        self.box_labels = box_labels
        self.heat_map: np.ndarray | None = None
        self.histogram: np.ndarray | None = None

        if anomaly_map is not None:
            self.heat_map = superimpose_anomaly_map(anomaly_map, image, normalize=True)
            if gt_mask is not None:
                self.histogram = _anomaly_score_histogram(anomaly_map, gt_mask)

        if gt_mask is not None and gt_mask.max() <= 1.0:
            self.gt_mask = (gt_mask * 255).astype(np.uint8)

        if pred_boxes is not None:
            if box_labels is None:
                raise ValueError("box_labels is required when pred_boxes is set.")
            self.normal_boxes = pred_boxes[~box_labels.astype(bool)]
            self.anomalous_boxes = pred_boxes[box_labels.astype(bool)]


def _anomaly_score_histogram(anomaly_map: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Render a normal-vs-anomaly score histogram matched to ``gt_mask`` spatial size."""
    map_2d = anomaly_map.squeeze()
    mask_bool = gt_mask.astype(bool)
    normal_scores = map_2d[~mask_bool].flatten()
    anomaly_scores = map_2d[mask_bool].flatten()

    num_bins = 100
    bins = np.linspace(map_2d.min(), map_2d.max(), num_bins)
    normal_hist, _ = np.histogram(normal_scores, bins=bins)
    anomaly_hist, _ = np.histogram(anomaly_scores, bins=bins)
    overlap_pct = np.minimum(
        normal_hist / max(normal_hist.sum(), 1),
        anomaly_hist / max(anomaly_hist.sum(), 1),
    ).sum() * 100

    plt.clf()
    normal_weights = np.ones_like(normal_scores) / max(len(normal_scores), 1)
    anomaly_weights = np.ones_like(anomaly_scores) / max(len(anomaly_scores), 1)
    normal_hist, _, _ = plt.hist(
        normal_scores, bins=bins, alpha=0.3, color="orange", label="Normal", weights=normal_weights
    )
    anomaly_hist, _, _ = plt.hist(
        anomaly_scores, bins=bins, alpha=0.3, color="green", label="Anomaly", weights=anomaly_weights
    )
    plt.plot([], [], " ", label=f"Overlap: {overlap_pct:.2f}%")
    plt.ylim(0, max(normal_hist.max(), anomaly_hist.max()) * 1.1)
    plt.legend(loc="upper right")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Relative Frequency")

    fig = plt.gcf()
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    histogram = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    histogram = histogram.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.resize(histogram, gt_mask.shape[::-1], interpolation=cv2.INTER_LINEAR)


class ImageVisualizer(BaseVisualizer):
    """Yield per-sample panels (RGB, GT mask, heatmap, histogram) during predict."""

    def __init__(
        self,
        task: TaskType = TaskType.SEGMENTATION,
        features_only: bool = False,
    ) -> None:
        super().__init__(VisualizationStep.BATCH)
        self.task = task
        self.features_only = features_only

    def generate(self, **kwargs) -> Iterator[tuple[GeneratorResult, ...]]:
        outputs = kwargs.get("outputs")
        if outputs is None:
            raise ValueError("outputs must be provided to generate images.")
        return self._visualize_batch(outputs)

    def _visualize_batch(self, batch: dict) -> Iterator[tuple[GeneratorResult, ...]]:
        if "image_path" not in batch:
            raise KeyError("Batch must contain 'image_path' for visualization.")

        batch_size = batch["image"].shape[0]
        for i in range(batch_size):
            height, width = batch["image"].shape[-2:]
            image = (read_image(path=batch["image_path"][i]) * 255).astype(np.uint8)
            image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
            file_name = Path(batch["image_path"][i])

            image_result = ImageResult(
                image=image,
                pred_score=_item(batch, "pred_scores", i),
                pred_label=_item(batch, "pred_labels", i),
                anomaly_map=_tensor_item(batch, "anomaly_maps", i),
                pred_mask=_tensor_item(batch, "pred_masks", i, squeeze=True),
                gt_mask=_tensor_item(batch, "mask", i, squeeze=True),
                gt_boxes=_tensor_item(batch, "boxes", i),
                pred_boxes=_tensor_item(batch, "pred_boxes", i),
                box_labels=_tensor_item(batch, "box_labels", i),
            )

            stem = str(file_name).replace(".png", "")
            yield (
                GeneratorResult(image=image_result.image, file_name=Path(f"{stem}_image.png")),
                GeneratorResult(image=image_result.gt_mask, file_name=Path(f"{stem}_mask.png")),
                GeneratorResult(image=image_result.heat_map, file_name=Path(f"{stem}_heatmap.png")),
                GeneratorResult(image=image_result.histogram, file_name=Path(f"{stem}_histogram.png")),
            )


def _item(batch: dict, key: str, index: int):
    if key not in batch:
        return None
    value = batch[key][index]
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy().item()
    return value


def _tensor_item(batch: dict, key: str, index: int, squeeze: bool = False):
    if key not in batch:
        return None
    value = batch[key][index].detach().cpu().numpy()
    if squeeze:
        value = np.squeeze(value)
    return value
