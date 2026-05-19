"""Base PL Module for Training Task."""


import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from jsonargparse import Namespace

import torch
from torch import nn, optim
import lightning.pytorch as pl
from lightning.pytorch.cli import instantiate_class, LRSchedulerTypeTuple
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.metrics import AnomalibMetricCollection
from anomalib.metrics.threshold import BaseThreshold

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback

from core.data import FusedTransforms as Transform
from core.utils import TaskType, LearningType


logger = logging.getLogger(__name__)


class BaseModule(pl.LightningModule, ABC):
    """BaseModule to train, validate, predict and test images.

    Acts as a base class for all the PL Modules in the library.
    """

    def __init__(self, crop_pad_on: bool = True, load_strict: bool = True) -> None:
        super().__init__()
        logger.info("Initializing %s model.", self.__class__.__name__)

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: nn.Module
        self.callbacks: list[Callback]

        self.image_threshold: BaseThreshold
        self.pixel_threshold: BaseThreshold

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

        self._task_type: str | None = None
        self._transform: Transform | None = None
        self._input_size: tuple[int, int] | None = None
        self._padding: tuple[int, int] = (0, 0)

        self._is_setup = False  #* flag to track if setup has been called from the trainer
        self.register_buffer("prev_loss", torch.tensor(0.0))

        self.crop_pad_on = crop_pad_on
        self.load_strict = load_strict

    @property
    def name(self) -> str:
        """Name of the model."""
        return self.__class__.__name__

    def setup(self, stage: str | None = None) -> None:
        """Calls the _setup method to build the model if the model is not already built."""
        if getattr(self, "model", None) is None or not self._is_setup:
            self._setup()
            if isinstance(stage, TrainerFn):
                #* only set the flag if the stage is a TrainerFn, which means the setup has been called from a trainer
                self._is_setup = True

    def _setup(self) -> None:
        """The _setup method is used to build the torch model dynamically or adjust something about them.

        The model implementer may override this method to build the model. This is useful when the model cannot be set
        in the `__init__` method because it requires some information or data that is not available at the time of
        initialization.
        """

    def forward(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> Any:  # noqa: ANN401
        """Perform the forward-pass by passing input tensor to the module.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Output tensor from the model.
        """
        del args, kwargs  #* These variables are not used.

        return self.model(batch)

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """To be implemented in the subclasses."""
        raise NotImplementedError("Validation step not implemented. Please implement `validation_step` in the subclass.")

    def predict_step(
        self,
        batch: dict[str, str | torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Step function called during :meth:`~lightning.pytorch.trainer.Trainer.predict`.

        By default, it calls :meth:`~lightning.pytorch.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Any): Current batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        del batch_idx, dataloader_idx  #* These variables are not used.

        batch = self.validation_step(batch)
        batch["image"] = self.crop_padding(batch["image"])
        return batch

    def test_step(self, batch: dict[str, str | torch.Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          batch_idx (int): Batch index
          args: Arguments.
          kwargs: Keyword arguments.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        return self.validation_step(batch, batch_idx, *args, **kwargs)

    def on_test_start(self) -> None:
        """Called when test begins."""
        if self.trainer.precision != "32-true":
            logger.warning("Test should be run with 32-bit precision for better performance.")

    @property
    @abstractmethod
    def trainer_arguments(self) -> dict[str, Any]:
        """Arguments used to override the trainer parameters so as to train the model correctly."""
        raise NotImplementedError

    def _save_to_state_dict(self, destination: OrderedDict, prefix: str, keep_vars: bool) -> None:
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> Any:  # noqa: ANN401
        """Initialize auxiliary objects if necessary."""
        if "prev_loss" not in state_dict:
            state_dict["prev_loss"] = torch.tensor(0.0)
        if not hasattr(self, "image_threshold"):
            #? Remove from state dict
            state_dict.pop("image_threshold.value", None)
        if not hasattr(self, "pixel_threshold"):
            #? Remove from state dict
            state_dict.pop("pixel_threshold.value", None)
        return super().load_state_dict(state_dict, self.load_strict)

    @property
    @abstractmethod
    def learning_type(self) -> LearningType:
        """Learning type of the model."""
        raise NotImplementedError

    @property
    def task_type(self) -> str:
        """Task type of the model."""
        if self._task_type is None:
            raise ValueError("Task type not set. Please set the task type of model before training.")
        return self._task_type

    @task_type.setter
    def task_type(self, task_type: str) -> None:
        """Set the task type of the model."""
        self._task_type = task_type

    @property
    def transform(self) -> Transform:
        """Retrieve the model-specific transform.

        If a transform has been set using `set_transform`, it will be returned.
        """
        return self._transform

    def set_transform(self, transform: Transform) -> None:
        """Update the transform linked to the model instance.
        Then update the input size of the model based on the transform."""
        self._transform = transform
        self._set_input_size()

    def configure_transforms(self, image_size: tuple[int, int] | None = None) -> Transform:
        """Default transforms. Removed from the model for more flexibility and control."""
        raise NotImplementedError("Configuring transforms within model is not supported.")

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Return the effective input size of the model."""
        return self._input_size

    @property
    def padding(self) -> tuple[int, int]:
        """Return the amount of padding added to the input image [H, W].
        Assumes center padding."""
        return self._padding
    
    def _set_input_size(self) -> None:
        """Set the input size of the model based on the transform."""
        if self._transform is None or self._transform.image_size is None:
            self._input_size = None
        else:
            dummy_input = torch.zeros((3, *self._transform.image_size))
            output_shape = self.transform(dummy_input).shape[-2:]
            self._input_size = output_shape
            self._set_padding()

    def _set_padding(self) -> None:
        """Set the padding on input based on transform."""
        if all([self._transform, self._transform.image_size, self._input_size]):
            self._padding = (
                (self._input_size[0] - self._transform.image_size[0]) // 2,
                (self._input_size[1] - self._transform.image_size[1]) // 2
            )

    def crop_padding(self, x: torch.Tensor, ori_shape: tuple[int, int]) -> torch.Tensor:
        """Crop the padding from the input tensor."""
        if not self.crop_pad_on:
            return x

        padding = (x.shape[-2] - ori_shape[0]) // 2, (x.shape[-1] - ori_shape[1]) // 2
        if padding == (0, 0):
            return x
        if padding[0] > 0:
            x = x[..., padding[0]:-padding[0], :]
        if padding[1] > 0:
            x = x[..., :, padding[1]:-padding[1]]
        return x

    def set_optimizer_scheduler_cfg(self, 
                                    optimizer_cfg: Namespace | None,
                                    lr_scheduler_cfg: Namespace | None) -> None:
        """Set optimizer and scheduler configuration.

        Args:
            optimizer_cfg (Namespace | None): Optimizer configuration.
            lr_scheduler_cfg (Namespace | None): Learning rate scheduler configuration.
        """
        self._optimizer_cfg = optimizer_cfg
        self._lr_scheduler_cfg = lr_scheduler_cfg

    @abstractmethod
    def param_groups(self) -> list[dict[str, Any]]:
        """Return the parameter groups for the optimizer."""
        raise NotImplementedError("Please implement `param_groups` in the subclass.")

    def configure_optimizers(self) -> tuple[list[optim.Optimizer], list[LRSchedulerTypeTuple]]:
        """Configure optimizer and learning rate scheduler.
        Some tips:
            - total_steps=self.trainer.estimated_stepping_batches
            - return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        """
        param_groups = self.param_groups()
        optimizer = instantiate_class(param_groups, self.optimizer_cfg)

        if self._lr_scheduler_cfg is not None and len(dict(self._lr_scheduler_cfg)) > 0:
            if hasattr(self._lr_scheduler_cfg.init_args, "total_steps"):
                self._lr_scheduler_cfg.init_args.total_steps = self.trainer.estimated_stepping_batches
            if hasattr(self._lr_scheduler_cfg.init_args, "total_iters"):
                self._lr_scheduler_cfg.init_args.total_iters = self.trainer.estimated_stepping_batches
            scheduler = instantiate_class(optimizer, self._lr_scheduler_cfg)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step"
            }
            return [optimizer], [lr_scheduler]
        return [optimizer]
    
    @property
    def optimizer_cfg(self) -> Namespace | None:
        """Optimizer configuration."""
        return self._optimizer_cfg
    
    @property
    def lr_scheduler_cfg(self) -> Namespace | None:
        """Learning rate scheduler configuration."""
        return self._lr_scheduler_cfg

    @property
    def score_range(self) -> tuple[float, float]:
        """Return the score range of the model.
        To be implemented in the subclasses."""
        raise NotImplementedError("Please override `score_range` in the subclass.")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when saving the model to a checkpoint.

        Saves the transform to the checkpoint.
        """
        checkpoint["transform"] = self.transform

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when loading the model from a checkpoint.

        Loads the transform from the checkpoint and calls setup to ensure that the torch model is built before loading
        the state dict.
        """
        self._transform = checkpoint["transform"]
        self.setup("load_checkpoint")
