"""Fit image- and pixel-level decision thresholds during validation."""

from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, ListConfig

from anomalib.metrics.threshold import BaseThreshold
from anomalib.utils.types import THRESHOLD

from core.models import BaseModule
from core.utils import UnknownModuleError


class ThresholdCallback(Callback):
    """Attach threshold objects to the module and update them from validation outputs."""

    def __init__(self, threshold: THRESHOLD = "F1Threshold") -> None:
        super().__init__()
        self._initialize_thresholds(threshold)
        self.state = {
            "image_threshold": self.image_threshold.value,
            "pixel_threshold": self.pixel_threshold.value,
        }

    def setup(self, trainer: Trainer, pl_module: BaseModule, stage: str) -> None:
        del trainer, stage
        if not hasattr(pl_module, "image_threshold"):
            pl_module.image_threshold = self.image_threshold
        if not hasattr(pl_module, "pixel_threshold"):
            pl_module.pixel_threshold = self.pixel_threshold

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: BaseModule) -> None:
        del trainer
        pl_module.image_threshold.reset()
        pl_module.pixel_threshold.reset()

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
            self._update(pl_module, outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseModule) -> None:
        del trainer
        self.state["image_threshold"] = pl_module.image_threshold.compute()
        self.state["pixel_threshold"] = pl_module.pixel_threshold.compute()

    def _initialize_thresholds(self, threshold: THRESHOLD) -> None:
        if isinstance(threshold, BaseThreshold):
            self.image_threshold = threshold
            self.pixel_threshold = threshold.clone()
        elif isinstance(threshold, tuple) and isinstance(threshold[0], BaseThreshold):
            self.image_threshold = threshold[0]
            self.pixel_threshold = threshold[1]
        elif isinstance(threshold, str | DictConfig | ListConfig | list):
            self._load_from_config(threshold)
        else:
            raise TypeError(f"Invalid threshold type: {type(threshold)}")

    def _load_from_config(
        self,
        threshold: DictConfig | str | ListConfig | list[dict[str, str | float]],
    ) -> None:
        if isinstance(threshold, str | DictConfig):
            self.image_threshold = self._get_threshold_from_config(threshold)
            self.pixel_threshold = self.image_threshold.clone()
        elif isinstance(threshold, ListConfig | list):
            self.image_threshold = self._get_threshold_from_config(threshold[0])
            self.pixel_threshold = self._get_threshold_from_config(threshold[1])
        else:
            raise TypeError(f"Invalid threshold config: {threshold}")

    def _get_threshold_from_config(
        self,
        threshold: DictConfig | str | dict[str, str | float],
    ) -> BaseThreshold:
        if isinstance(threshold, str):
            threshold = DictConfig({"class_path": threshold})

        class_path = threshold["class_path"]
        init_args = threshold.get("init_args", {})

        if len(class_path.split(".")) == 1:
            module_path = "core.utils.metrics.threshold"
            class_name = class_path
        else:
            module_path = ".".join(class_path.split(".")[:-1])
            class_name = class_path.split(".")[-1]

        try:
            module = __import__(module_path, fromlist=[class_name])
        except ImportError as exc:
            raise UnknownModuleError(f"Module {module_path} not found.") from exc
        try:
            class_ = getattr(module, class_name)
        except AttributeError as exc:
            raise UnknownModuleError(
                f"Class {class_name} not found in module {module_path}."
            ) from exc

        return class_(**init_args)

    def _update(self, pl_module: BaseModule, outputs: STEP_OUTPUT) -> None:
        pl_module.image_threshold.update(outputs["pred_scores"], outputs["label"].int())
        if "mask" in outputs and "anomaly_maps" in outputs:
            pl_module.pixel_threshold.update(
                torch.squeeze(outputs["anomaly_maps"]),
                torch.squeeze(outputs["mask"]).int(),
            )

    def load_state_dict(self, state_dict: dict) -> None:
        self.state.update(state_dict)

    def state_dict(self) -> dict:
        return self.state.copy()

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        checkpoint: dict,
    ) -> None:
        del trainer, pl_module
        checkpoint["state_dict"]["image_threshold.value"] = self.state["image_threshold"]
        checkpoint["state_dict"]["pixel_threshold.value"] = self.state["pixel_threshold"]
