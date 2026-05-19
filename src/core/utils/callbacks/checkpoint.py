"""Model checkpoint callback with zero-/few-shot validation support."""

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as LightningCheckpoint
from lightning.pytorch.trainer.states import TrainerFn

from core.utils import LearningType


class ModelCheckpoint(LightningCheckpoint):
    """Lightning checkpoint that can save during validation-only (zero-/few-shot) runs.

    Overrides when checkpoints are skipped and whether they align with train or validation
    epoch ends for models that never run a training loop.
    """

    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        if hasattr(trainer.model, "module"):
            model = trainer.model.module
        else:
            model = trainer.model
        is_zero_or_few_shot = model.learning_type in (LearningType.ZERO_SHOT, LearningType.FEW_SHOT)
        return (
            bool(trainer.fast_dev_run)
            or trainer.state.fn not in (TrainerFn.FITTING, TrainerFn.VALIDATING)
            or trainer.sanity_checking
            or (self._last_global_step_saved == trainer.global_step and not is_zero_or_few_shot)
        )

    def _should_save_on_train_epoch_end(self, trainer: Trainer) -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

        if hasattr(trainer.model, "module"):
            model = trainer.model.module
        else:
            model = trainer.model
        if model.learning_type in (LearningType.ZERO_SHOT, LearningType.FEW_SHOT):
            return False

        return super()._should_save_on_train_epoch_end(trainer)
