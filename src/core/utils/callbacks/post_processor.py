"""Post-process model outputs: sigmoid scaling, derived scores, and thresholded labels."""

from typing import Any

import torch
from lightning import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from wandb import AlertLevel

import normflows as nf

from core.models import BaseModule
from core.utils.loggers.wandb import CustomWandbLogger


class PostProcessorCallback(Callback):
    """Normalize scores, derive missing fields, and apply thresholds on val/test/predict batches."""

    def __init__(self, apply_sigmoid: bool = True) -> None:
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

    def on_train_epoch_start(self, trainer: Trainer, pl_module: BaseModule) -> None:
        if hasattr(pl_module, "synthesize"):
            trainer.datamodule.train_transform.synthesize = pl_module.synthesize

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx, outputs

        if hasattr(pl_module.model, "flow_blocks") and getattr(pl_module, "flow_type", None) == "residual":
            for nfm in pl_module.model.flow_blocks:
                nf.utils.update_lipschitz(nfm, 10)

        if hasattr(pl_module, "prev_loss") and (
            torch.isnan(pl_module.prev_loss) or torch.isinf(pl_module.prev_loss)
        ):
            for logger in trainer.loggers:
                if isinstance(logger, CustomWandbLogger):
                    logger.experiment.alert(
                        title="Training Loss NaN or Inf",
                        text=f"Loss is NaN or Inf at step {trainer.global_step}.",
                        level=AlertLevel.ERROR,
                        wait_duration=60 * 30,
                    )
            raise AssertionError("Loss is NaN or Inf.")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx
        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx
        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        outputs: Any,  # noqa: ANN401
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx
        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def post_process(self, trainer: Trainer, pl_module: BaseModule, outputs: STEP_OUTPUT) -> None:
        if not isinstance(outputs, dict):
            return
        if self.apply_sigmoid:
            self._normalize_batch(pl_module, outputs)
        self._post_process(outputs)
        if trainer.predicting:
            self._compute_scores_and_labels(pl_module, outputs)

    @staticmethod
    def _compute_scores_and_labels(pl_module: BaseModule, outputs: STEP_OUTPUT) -> None:
        if "pred_scores" in outputs and hasattr(pl_module, "image_threshold"):
            threshold = pl_module.image_threshold
            value = threshold if isinstance(threshold, float) else threshold.value
            outputs["pred_labels"] = outputs["pred_scores"] >= value
        if "anomaly_maps" in outputs and hasattr(pl_module, "pixel_threshold"):
            threshold = pl_module.pixel_threshold
            value = threshold if isinstance(threshold, float) else threshold.value
            outputs["pred_masks"] = outputs["anomaly_maps"] >= value

    @staticmethod
    def _post_process(outputs: STEP_OUTPUT) -> None:
        if "pred_scores" not in outputs and "anomaly_maps" in outputs:
            outputs["pred_scores"] = (
                outputs["anomaly_maps"]
                .reshape(outputs["anomaly_maps"].shape[0], -1)
                .max(dim=1)
                .values
            )

    @staticmethod
    def _normalize_batch(pl_module: BaseModule, outputs: STEP_OUTPUT) -> None:
        if not hasattr(pl_module, "score_range"):
            raise AttributeError("Model must define `score_range` for score normalization.")

        min_ = (
            torch.tensor(pl_module.score_range[0]).sigmoid()
            if pl_module.score_range[0] is not None
            else 0.0
        )
        max_ = (
            torch.tensor(pl_module.score_range[1]).sigmoid()
            if pl_module.score_range[1] is not None
            else 1.0
        )

        for key in ("anomaly_maps", "pred_scores"):
            if key not in outputs:
                continue
            if torch.all((outputs[key] >= 0) & (outputs[key] <= 1)):
                continue
            outputs[key] = outputs[key].sigmoid()
            outputs[key] = (outputs[key] - min_) / (max_ - min_)
