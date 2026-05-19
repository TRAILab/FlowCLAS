"""Checkpoint callback that also persists a companion ``-EMA`` weights file."""

import os.path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_info

from .ema import EMA


class EMAModelCheckpoint(ModelCheckpoint):
    """Save standard checkpoints plus an EMA copy when an :class:`EMA` callback is registered."""

    def _get_ema_callback(self, trainer: pl.Trainer) -> EMA | None:
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                return callback
        return None

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is None:
            return

        ema_callback.replace_model_weights(trainer.lightning_module)
        ema_path = self._ema_format_filepath(filepath)
        if self.verbose:
            rank_zero_info(f"Saving EMA weights to {ema_path}")
        super()._save_checkpoint(trainer, ema_path)
        ema_callback.restore_original_weights(trainer.lightning_module)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, torch.Tensor],
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            op = max if self.mode == "min" else min
            self.kth_best_model_path = op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        op = min if self.mode == "min" else max
        self.best_model_path = op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f} "
                f"(best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        if del_filepath is not None and filepath != del_filepath:
            self._remove_checkpoint(trainer, del_filepath)
            self._remove_checkpoint(
                trainer,
                del_filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}"),
            )
