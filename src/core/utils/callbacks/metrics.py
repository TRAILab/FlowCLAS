"""Attach metric collections to the Lightning module and log epoch results."""

import logging
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.metrics import AnomalibMetricCollection

from core.models import BaseModule
from core.utils import Device, TaskType
from core.utils.metrics import BaseMetric, create_metric_collection

logger = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """Create image/pixel metric collections on the module and update them each val/test step."""

    def __init__(
        self,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_metrics: list[BaseMetric] | list[str] | str | None = None,
        pixel_metrics: list[BaseMetric] | list[str] | str | None = None,
        sigmoid_applied: bool = True,
        device: Device = Device.GPU,
    ) -> None:
        super().__init__()
        self.task = TaskType(task)
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.device = device
        self.sigmoid_applied = sigmoid_applied
        self.check_image_output = False
        self.check_pixel_output = False

    def setup(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        stage: str | None = None,
    ) -> None:
        image_metric_names = [] if self.image_metric_names is None else self.image_metric_names
        if isinstance(image_metric_names, str):
            image_metric_names = [image_metric_names]

        if self.pixel_metric_names is None:
            pixel_metric_names: list[str] = []
        elif self.task == TaskType.CLASSIFICATION:
            pixel_metric_names = []
            logger.warning(
                "Ignoring pixel metrics for classification task: %s",
                self.pixel_metric_names,
            )
        else:
            pixel_metric_names = (
                self.pixel_metric_names
                if not isinstance(self.pixel_metric_names, str)
                else [self.pixel_metric_names]
            )

        if isinstance(pl_module, BaseModule):
            pl_module.image_metrics = create_metric_collection(image_metric_names, "i_")
            pl_module.pixel_metrics = create_metric_collection(pixel_metric_names, "p_")

            if stage != "predict":
                for metric in pl_module.image_metrics._modules.values():
                    if getattr(metric, "thresholds", None) is not None:
                        self.check_image_output = True
                        break
                for metric in pl_module.pixel_metrics._modules.values():
                    if getattr(metric, "thresholds", None) is not None:
                        self.check_pixel_output = True
                        break
                if self.check_image_output or self.check_pixel_output:
                    assert self.sigmoid_applied, (
                        "Threshold-binned metrics require sigmoid-normalized scores in [0, 1]."
                    )

        self._persist_off(pl_module)
        self.prev_dataloader_idx = 0
        self.prefixes = getattr(trainer.datamodule, "data_names", [""])
        self.image_metric_logs: list[dict] = []
        self.pixel_metric_logs: list[dict] = []

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: BaseModule) -> None:
        del trainer
        pl_module.image_metrics.reset()
        pl_module.pixel_metrics.reset()

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, batch, batch_idx

        if self.prev_dataloader_idx != dataloader_idx:
            image_metrics = pl_module.image_metrics.to(Device.GPU)
            pixel_metrics = pl_module.pixel_metrics.to(Device.GPU)
            self.image_metric_logs.append(image_metrics.compute())
            self.pixel_metric_logs.append(pixel_metrics.compute())
            self.prev_dataloader_idx = dataloader_idx
            image_metrics.reset()
            pixel_metrics.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx
        if outputs is not None:
            self._outputs_to_device(outputs)
            self._move_metrics_to_device(pl_module, self.device)
            self._update_metrics(
                pl_module.image_metrics,
                pl_module.pixel_metrics,
                outputs,
                pl_module.task_type,
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseModule) -> None:
        del trainer
        self._move_metrics_to_device(pl_module, Device.GPU)
        self.image_metric_logs.append(pl_module.image_metrics.compute())
        self.pixel_metric_logs.append(pl_module.pixel_metrics.compute())
        self._log_metrics(pl_module)
        self.prev_dataloader_idx = 0
        self.image_metric_logs = []
        self.pixel_metric_logs = []

    def on_test_epoch_start(self, trainer: Trainer, pl_module: BaseModule) -> None:
        self.on_validation_epoch_start(trainer, pl_module)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: BaseModule) -> None:
        self.on_validation_epoch_end(trainer, pl_module)

    def _update_metrics(
        self,
        image_metric: AnomalibMetricCollection,
        pixel_metric: AnomalibMetricCollection,
        output: STEP_OUTPUT,
        task: TaskType,
    ) -> None:
        if task == TaskType.RECONSTRUCTION:
            if "pred_mask" in output:
                image_metric.update(
                    output["pred"],
                    output["target"],
                    output["mask"].int(),
                    output["pred_mask"].int(),
                )
            else:
                image_metric.update(output["pred"], output["target"], output["mask"].int())
        elif task == TaskType.PRETRAIN:
            image_metric.update(output["pred"], output["mask"].int())
        else:
            if self.check_image_output and not torch.all(
                (output["pred_scores"] >= 0) & (output["pred_scores"] <= 1)
            ):
                raise ValueError(
                    "Image scores must lie in [0, 1] for threshold-binned metrics; "
                    f"got min={output['pred_scores'].min()}, max={output['pred_scores'].max()}."
                )
            image_metric.update(output["pred_scores"], output["label"].int())

        if task == TaskType.SEGMENTATION and "mask" in output and "anomaly_maps" in output:
            if self.check_pixel_output and not torch.all(
                (output["anomaly_maps"] >= 0) & (output["anomaly_maps"] <= 1)
            ):
                raise ValueError("Pixel anomaly maps must lie in [0, 1] for threshold-binned metrics.")
            pixel_metric.update(
                torch.squeeze(output["anomaly_maps"]),
                torch.squeeze(output["mask"].int()),
            )

    def _outputs_to_device(self, output: STEP_OUTPUT) -> STEP_OUTPUT | dict[str, Any]:
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_device(value)
        elif isinstance(output, torch.Tensor) and output.device != self.device:
            output = output.to(self.device)
        return output

    @staticmethod
    def _persist_off(pl_module: BaseModule) -> None:
        pl_module.image_metrics.persistent(False)
        pl_module.pixel_metrics.persistent(False)

    @staticmethod
    def _move_metrics_to_device(pl_module: BaseModule, device: Device) -> None:
        pl_module.image_metrics.to(device)
        pl_module.pixel_metrics.to(device)

    def _log_metrics(self, pl_module: BaseModule) -> None:
        if pl_module.pixel_metrics._update_called:  # noqa: SLF001
            for prefix, i_log_dict, p_log_dict in zip(
                self.prefixes,
                self.image_metric_logs,
                self.pixel_metric_logs,
            ):
                suffix = f"-{prefix}" if prefix else ""
                pl_module.log_dict(
                    {f"{key}{suffix}": value for key, value in p_log_dict.items()},
                    sync_dist=True,
                    prog_bar=True,
                )
                pl_module.log_dict(
                    {f"{key}{suffix}": value for key, value in i_log_dict.items()},
                    sync_dist=True,
                    prog_bar=False,
                )
        else:
            for prefix, log_dict in zip(self.prefixes, self.image_metric_logs):
                suffix = f"-{prefix}" if prefix else ""
                pl_module.log_dict(
                    {f"{key}{suffix}": value for key, value in log_dict.items()},
                    sync_dist=True,
                    prog_bar=True,
                )
