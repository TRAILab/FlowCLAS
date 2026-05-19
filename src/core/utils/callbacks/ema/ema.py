"""Exponential moving average (EMA) of model weights during training."""

import os
import warnings
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT

# Adapted from NVIDIA NeMo EMA callback.


class EMA(Callback):
    """Maintain EMA weight copies; optionally validate and checkpoint against EMA weights."""

    def __init__(
        self,
        decay: float = 0.998,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = True,
    ) -> None:
        if not 0 <= decay <= 1:
            raise MisconfigurationException("EMA decay must be between 0 and 1.")
        self.decay = decay
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self._ema_model_weights: list[torch.Tensor] | None = None
        self._overflow_buf: torch.Tensor | None = None
        self._cur_step: int | None = None
        self._weights_buffer: list[torch.Tensor] | None = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        rank_zero_info("Creating EMA weights copy.")
        if self._ema_model_weights is None:
            self._ema_model_weights = [p.detach().clone() for p in pl_module.model.state_dict().values()]
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: pl.LightningModule) -> None:
        for ema_w, (_name, model_w) in zip(self._ema_model_weights, pl_module.model.state_dict().items()):
            if ema_w.dtype not in (torch.long, torch.bool) and model_w.dtype not in (torch.long, torch.bool):
                ema_w.mul_(self.decay).add_(model_w.detach(), alpha=1 - self.decay)

    def should_apply_ema(self, step: int) -> bool:
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> dict[str, Any]:
        if self.save_ema_weights_in_callback_state:
            return {"cur_step": self._cur_step, "ema_weights": self._ema_model_weights}
        return {"cur_step": self._cur_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cur_step = state_dict["cur_step"]
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del pl_module, checkpoint
        checkpoint_callback = trainer.checkpoint_callback
        if not (trainer.ckpt_path and checkpoint_callback is not None):
            return

        ckpt_path = str(trainer.ckpt_path)
        ext = checkpoint_callback.FILE_EXTENSION
        if ckpt_path.endswith(f"-EMA{ext}"):
            rank_zero_info("Loaded EMA checkpoint; a fresh EMA copy will be created when training resumes.")
            return

        ema_path = ckpt_path.replace(ext, f"-EMA{ext}")
        if os.path.exists(ema_path):
            ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
            self._ema_model_weights = list(ema_state_dict["state_dict"].values())
            del ema_state_dict
            rank_zero_info("Restored EMA weights from companion checkpoint.")
        else:
            warnings.warn(
                "No companion EMA checkpoint found; training will start with a new EMA copy.",
                stacklevel=2,
            )

    def replace_model_weights(self, pl_module: pl.LightningModule) -> None:
        self._weights_buffer = [p.detach().clone().to("cpu") for p in pl_module.model.state_dict().values()]
        new_state_dict = dict(zip(pl_module.model.state_dict().keys(), self._ema_model_weights))
        pl_module.model.load_state_dict(new_state_dict, strict=getattr(pl_module, "load_strict", True))

    def restore_original_weights(self, pl_module: pl.LightningModule) -> None:
        new_state_dict = dict(zip(pl_module.model.state_dict().keys(), self._weights_buffer))
        pl_module.model.load_state_dict(new_state_dict, strict=getattr(pl_module, "load_strict", True))
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)
