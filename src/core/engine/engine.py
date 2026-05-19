"""Custom training engine wrapping PyTorch Lightning's Trainer (Anomalib-style)."""

import datetime
import logging
import os.path as osp
from pathlib import Path

from jsonargparse import Namespace

import torch
from lightning.pytorch.callbacks import (
    Callback, EarlyStopping, LearningRateMonitor, TQDMProgressBar
)
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from anomalib.callbacks.timer import TimerCallback
from anomalib.data import AnomalibDataset, PredictDataset
from anomalib.engine.engine import (
    _TrainerArgumentsCache as TrainerArgumentsCache,
    UnassignedError
)
from anomalib.utils.path import create_versioned_dir
from anomalib.utils.types import THRESHOLD

from core.data import BaseDataModule
from core.models import BaseModule
from core.utils import Device, LearningType, TaskType
from core.utils.callbacks import (
    EMA,
    EMAModelCheckpoint,
    MetricsCallback,
    ModelCheckpoint,
    PostProcessorCallback,
    ThresholdCallback,
)
from core.utils.callbacks import get_visualizer_callback
from core.utils.loggers import get_experiment_logger
from core.utils.metrics import BaseMetric

logger = logging.getLogger(__name__)

# TF32 on Ampere+ (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class CustomEngine:
    """Configure and run Lightning training, testing, and prediction for ``BaseModule`` models.

    Additional keyword arguments are forwarded to :class:`~lightning.pytorch.trainer.trainer.Trainer`.

    Args:
        callbacks: Extra Lightning callbacks merged into the trainer cache before engine callbacks are prepended.
        threshold: Thresholding method name or config (Anomalib ``THRESHOLD``).
        task: Segmentation or classification task type.
        image_metrics: Names or metric objects for image-level evaluation.
        pixel_metrics: Names or metric objects for pixel-level evaluation (segmentation).
        metrics_on_cpu: If True, run metric computation on CPU.
        default_root_dir: Root directory for checkpoints and logs (before experiment subpath).
        experiment_cfg: Experiment name, logging, and versioning options.
        visualization: Settings for optional visualization callbacks.
        monitor1: Primary checkpoint metric (used when no ``EarlyStopping`` callback supplies ``monitor``).
        monitor2: Optional second checkpoint metric group.
        save_every_epoch: If True, save a checkpoint every training epoch.
        apply_sigmoid: Passed to ``PostProcessorCallback`` and ``MetricsCallback``.
        ema: If True, use EMA variants of checkpoints and append an ``EMA`` callback.
        **kwargs: Lightning ``Trainer`` constructor arguments (e.g. ``max_epochs``, ``callbacks``).
    """

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        threshold: THRESHOLD = "F1AdaptiveThreshold",
        task: TaskType | str = TaskType.SEGMENTATION,
        image_metrics: str | list[str] | list[BaseMetric] | None = None,
        pixel_metrics: str | list[str] | list[BaseMetric] | None = None,
        metrics_on_cpu: bool = False,
        default_root_dir: str | Path = "../results",
        experiment_cfg: Namespace | None = None,
        visualization: Namespace | None = None,
        monitor1: Namespace | None = None,
        monitor2: Namespace | None = None,
        save_every_epoch: bool = False,
        apply_sigmoid: bool = True,
        ema: bool = False,
        **kwargs,
    ) -> None:
        if callbacks is None:
            callbacks = []

        # Cache Lightning Trainer constructor arguments.
        self._cache = TrainerArgumentsCache(
            callbacks=[*callbacks],
            default_root_dir=Path(default_root_dir),
            **kwargs,
        )
        self.experiment_cfg = experiment_cfg
        self.visualization_cfg = visualization
        self.threshold = threshold
        self.task = TaskType(task)
        self.save_every_epoch = save_every_epoch

        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.metrics_on_cpu = metrics_on_cpu
        self.apply_sigmoid = apply_sigmoid
        self.ema = ema

        self.monitor1 = monitor1
        self.monitor2 = monitor2

        self._trainer: Trainer | None = None

    @property
    def trainer(self) -> Trainer:
        """The bound Lightning ``Trainer`` instance.

        Raises:
            UnassignedError: If no trainer has been created yet.
        """
        if not self._trainer:
            msg = "``self.trainer`` is not assigned yet."
            raise UnassignedError(msg)
        return self._trainer

    @property
    def model(self) -> BaseModule:
        """The active ``BaseModule`` (``trainer.lightning_module``).

        Raises:
            UnassignedError: If the trainer has no module attached.
        """
        if not self.trainer.model:
            msg = "Trainer does not have a model assigned yet."
            raise UnassignedError(msg)
        return self.trainer.lightning_module

    @property
    def post_process_callback(self) -> PostProcessorCallback | None:
        """Single ``PostProcessorCallback`` from ``trainer.callbacks``, if exactly one exists.

        Raises:
            ValueError: If more than one post-processor callback is registered.
        """
        callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, PostProcessorCallback)]
        if len(callbacks) > 1:
            msg = (
                f"Trainer can only have one post-processing callback but multiple found: {callbacks}. "
                "Please check your configuration. Exiting to avoid unexpected behavior."
            )
            raise ValueError(msg)
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def threshold_callback(self) -> ThresholdCallback | None:
        """Single ``ThresholdCallback`` from ``trainer.callbacks``, if exactly one exists.

        Raises:
            ValueError: If more than one threshold callback is registered.
        """
        callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, ThresholdCallback)]
        if len(callbacks) > 1:
            msg = (
                f"Trainer can only have one thresholding callback but multiple found: {callbacks}. "
                "Please check your configuration. Exiting to avoid unexpected behavior."
            )
            raise ValueError(msg)
        return callbacks[0] if len(callbacks) > 0 else None

    def _setup_experiment(
        self,
        log: bool = True,
        versioned_dir: bool = True,
    ) -> None:
        """Configure artifact root, optional versioning, and Lightning loggers.

        Args:
            log: When ``False``, set ``logger`` to ``False`` (no experiment loggers).
            versioned_dir: When ``True``, create a new version folder unless ``latest`` is recent enough.
        """
        # Resolve experiment output directory (optionally versioned by time window).
        root_dir = Path(self._cache.args["default_root_dir"]) / self.experiment_cfg.name
        latest_dir = root_dir / "latest"
        if latest_dir.exists():
            # Reuse ``latest`` only if it was created within ``experiment_cfg.time_delta`` (DDP-friendly).
            creation_time = datetime.datetime.fromtimestamp(osp.getctime(root_dir / "latest"))
            current_time = datetime.datetime.now()
            time_delta = current_time - creation_time
            if time_delta.total_seconds() <= self.experiment_cfg.time_delta:
                versioned_dir = False

        self._cache.args["default_root_dir"] = create_versioned_dir(root_dir) if versioned_dir else root_dir / "latest"

        self._cache.args["logger"] = get_experiment_logger(self.experiment_cfg,
                                                           self._cache.args["default_root_dir"]) if log else log

    def _setup_trainer(self, model: BaseModule) -> None:
        """Build or refresh the Lightning ``Trainer`` from cached arguments and model-derived settings."""
        if self._cache.requires_update(model):
            self._cache.update(model)

        model.task_type = self.task

        self._setup_anomalib_callbacks()

        if self._trainer is None:
            self._trainer = Trainer(**self._cache.args)

    def _setup_dataset_task(
        self,
        *dataloaders: EVAL_DATALOADERS | TRAIN_DATALOADERS | BaseDataModule | None,
    ) -> None:
        """Set ``task`` on any ``BaseDataModule`` datasets to match this engine's ``task``."""
        for dataloader in dataloaders:
            if dataloader is not None and isinstance(dataloader, BaseDataModule):
                for attribute in ("train_data", "val_data", "test_data"):
                    if hasattr(dataloader, attribute):
                        data: AnomalibDataset = getattr(dataloader, attribute)
                        if data.task != self.task:
                            logger.info(
                                f"Overriding task from {data.task} with {self.task} for {dataloader.__class__}",
                            )
                            data.task = self.task

    @staticmethod
    def _setup_transform(
        model: BaseModule,
        datamodule: BaseDataModule | None = None,
        dataloaders: EVAL_DATALOADERS | TRAIN_DATALOADERS | None = None,
        ckpt_path: Path | str | None = None,
    ) -> None:
        """Resolve where to load the input transform from and assign it on ``model``.

        Precedence: datamodule ``transform``, else first dataloader dataset ``transform``, else ``transform`` key in
        ``ckpt_path`` checkpoint.

        Args:
            model: Module receiving ``set_transform``.
            datamodule: Optional data module whose ``transform`` is used when set.
            dataloaders: Optional loader(s) whose dataset ``transform`` may be used.
            ckpt_path: Optional checkpoint path containing a ``transform`` entry.
        """
        if isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]

        if datamodule and datamodule.transform:
            # Explicit datamodule transform wins.
            logger.info("Using transform from the datamodule.")
            transform = datamodule.transform
        elif dataloaders and any(getattr(dl.dataset, "transform", None) for dl in dataloaders):
            logger.info("Using transform from the first dataloader that has a transform.")
            transform = next(dl.dataset.transform for dl in dataloaders if getattr(dl.dataset, "transform", None))
        elif ckpt_path is not None:
            logger.info("Loading transform from checkpoint: %s", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=model.device)
            transform = checkpoint["transform"]
        else:
            raise ValueError("No transform provided and no transform found in the dataloaders or datamodule.")

        model.set_transform(transform)

    def _setup_anomalib_callbacks(self) -> None:
        """Rebuild core callbacks (checkpoints, metrics, visualization) and prepend them to the trainer list."""
        _callbacks: list[Callback] = []

        # Drop any user ``ModelCheckpoint`` instances; we attach our own policy below.
        _callbacks.extend([c for c in self._cache.args["callbacks"] if not isinstance(c, ModelCheckpoint)])

        early_stopping_callback = [c for c in self._cache.args["callbacks"] if isinstance(c, EarlyStopping)]
        has_early_stopping = len(early_stopping_callback) > 0
        monitor_metric = early_stopping_callback[0].monitor if has_early_stopping else self.monitor1.metric
        monitor_mode = early_stopping_callback[0].mode if has_early_stopping else self.monitor1.mode

        monitor_metric = [monitor_metric] if not isinstance(monitor_metric, list) else monitor_metric
        checkpoint_cb = EMAModelCheckpoint if self.ema else ModelCheckpoint
        for metric in monitor_metric:
            _callbacks.append(
                checkpoint_cb(
                    dirpath=self._cache.args["default_root_dir"] / "weights",
                    filename=f"best_{metric}_epoch_{{epoch}}_step_{{step}}",
                    save_top_k=1,
                    monitor=metric,
                    mode=monitor_mode,
                    auto_insert_metric_name=False,
                )
            )
        if self.monitor2.metric is not None:
            monitor2_metric = [self.monitor2.metric] if not isinstance(self.monitor2.metric, list) \
                                else self.monitor2.metric
            for metric in monitor2_metric:
                _callbacks.append(
                    checkpoint_cb(
                        dirpath=self._cache.args["default_root_dir"] / "weights",
                        filename=f"best_{metric}_epoch_{{epoch}}_step_{{step}}",
                        save_top_k=1,
                        monitor=metric,
                        mode=self.monitor2.mode,
                        auto_insert_metric_name=False,
                    ),
                )
        _callbacks.append(
            checkpoint_cb(
                dirpath=self._cache.args["default_root_dir"] / "weights",
                filename="last_epoch_{epoch}_step_{step}",
                save_top_k=1,
                save_on_train_epoch_end=True,
                monitor="step",
                mode="max",
                auto_insert_metric_name=False,
            ),
        )
        if self.save_every_epoch:
            _callbacks.append(
                ModelCheckpoint(
                    dirpath=self._cache.args["default_root_dir"] / "weights",
                    filename="epoch_{epoch}_step_{step}",
                    every_n_epochs=1,
                    save_on_train_epoch_end=True,
                    auto_insert_metric_name=False,
                ),
            )
        if self.ema:
            _callbacks.append(EMA())

        if self.task in (TaskType.CLASSIFICATION, TaskType.SEGMENTATION):
            _callbacks.append(ThresholdCallback(threshold=self.threshold))
            _callbacks.append(PostProcessorCallback(apply_sigmoid=self.apply_sigmoid))

        device = Device.CPU if self.metrics_on_cpu else Device.GPU
        _callbacks.append(MetricsCallback(self.task, 
                                          self.image_metric_names, 
                                          self.pixel_metric_names,
                                          device=device,
                                          sigmoid_applied=self.apply_sigmoid))

        if self.task in [TaskType.CLASSIFICATION, TaskType.SEGMENTATION]:
            _callbacks.append(
                get_visualizer_callback(
                    visualizers=self.visualization_cfg.visualizers,
                    save=self.visualization_cfg.save,
                    total=self.visualization_cfg.total,
                    features_only=self.visualization_cfg.features_only,
                    root=self._cache.args["default_root_dir"],
                    task=self.task
                )
            )

        _callbacks.extend([
            TimerCallback(),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=self._cache.args["log_every_n_steps"])])

        self._cache.args["callbacks"] = _callbacks + self._cache.args["callbacks"]

    def _should_run_validation(
        self,
        model: BaseModule,
        dataloaders: EVAL_DATALOADERS | None,
        datamodule: BaseDataModule | None,
        ckpt_path: str | Path | None,
    ) -> bool:
        """Return whether to run a short validation pass before test/predict.

        Used for zero- and few-shot flows when a ``ThresholdCallback`` is present, thresholds are not yet on the model,
        no checkpoint path is given, and validation data exists via ``dataloaders`` or ``datamodule``.

        Args:
            model: Active lightning module.
            dataloaders: Optional eval loader(s); ``None`` skips unless ``datamodule`` supplies val data.
            datamodule: Optional data module.
            ckpt_path: If set, validation is skipped (weights and state come from disk).

        Returns:
            ``True`` if the engine should call ``trainer.validate`` before test/predict.
        """
        if model.learning_type not in [LearningType.ZERO_SHOT, LearningType.FEW_SHOT]:
            return False
        if ckpt_path is not None:
            return False
        needs_thresholding = self.threshold_callback is not None \
                                and not (hasattr(model, "image_threshold") and hasattr(model, "pixel_threshold"))

        return needs_thresholding and (dataloaders is not None or datamodule is not None)

    def fit(
        self,
        model: BaseModule,
        train_dataloaders: TRAIN_DATALOADERS | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: BaseDataModule | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        """Train or, for zero-/few-shot models, run validation only (no ``fit`` loop).

        Args:
            model: Model to train or validate.
            train_dataloaders: Lightning train loader specification.
            val_dataloaders: Lightning validation loader specification.
            datamodule: Optional data module supplying loaders.
            ckpt_path: Optional resolved checkpoint path to restore weights before running.
        """
        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()

        self._setup_experiment()
        self._setup_trainer(model)
        self._setup_dataset_task(train_dataloaders, val_dataloaders, datamodule)
        self._setup_transform(model, datamodule=datamodule, ckpt_path=ckpt_path)
        if model.learning_type in [LearningType.ZERO_SHOT, LearningType.FEW_SHOT]:
            self.trainer.validate(model, val_dataloaders, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

    def test(
        self,
        model: BaseModule | None = None,
        dataloaders: EVAL_DATALOADERS | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: BaseDataModule | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Run ``trainer.test`` after optional setup and a pre-test validation pass when required.

        Args:
            model: Module to evaluate; may be omitted if already bound from a prior ``fit``.
            dataloaders: Test loader(s).
            ckpt_path: ``"best"``, ``"last"``, ``"hpc"``, or filesystem path to weights.
            verbose: Whether Lightning prints per-dataloader output.
            datamodule: Optional data module defining ``test_dataloader``.

        Returns:
            Per-dataloader metric dictionaries from Lightning.
        """
        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()

        self._setup_experiment(log=True)
        if model:
            self._setup_trainer(model)
        elif not self.model:
            msg = "`Engine.test()` requires a `BaseModule` when it hasn't been passed in a previous run."
            raise RuntimeError(msg)

        self._setup_dataset_task(dataloaders)
        self._setup_transform(model or self.model, datamodule=datamodule, ckpt_path=ckpt_path)
        if self._should_run_validation(model or self.model, dataloaders, datamodule, ckpt_path):
            logger.info("Running validation before testing (threshold / zero-shot setup).")
            self.trainer.validate(model, dataloaders, None, verbose=False, datamodule=datamodule)
        return self.trainer.test(model, dataloaders, ckpt_path, verbose, datamodule)

    def predict(
        self,
        model: BaseModule | None = None,
        dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: BaseDataModule | None = None,
        dataset: Dataset | PredictDataset | None = None,
        return_predictions: bool | None = False,
        ckpt_path: str | Path | None = None,
    ) -> _PREDICT_OUTPUT | None:
        """Run ``trainer.predict`` after optional setup and a pre-predict validation pass when required.

        Args:
            model: Module used for inference; may be omitted if already bound from a prior run.
            dataloaders: Predict loader(s).
            datamodule: Optional data module defining ``predict_dataloader``.
            dataset: When set, wrapped in a ``DataLoader`` and merged with ``dataloaders``.
            return_predictions: Forwarded to Lightning; default ``False``.
            ckpt_path: ``"best"``, ``"last"``, ``"hpc"``, or filesystem path to weights.

        Returns:
            Predictions or ``None`` depending on Lightning and ``return_predictions``.
        """
        assert (
            model or self.model
        ), "`Engine.predict()` requires a `BaseModule` when it hasn't been passed in a previous run."

        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()

        self._setup_experiment(log=False)
        if model:
            self._setup_trainer(model)

        if not ckpt_path:
            logger.warning("ckpt_path is not provided. Model weights will not be loaded.")

        if dataset is not None:
            dataloader = DataLoader(dataset)
            if dataloaders is None:
                dataloaders = dataloader
            elif isinstance(dataloaders, DataLoader):
                dataloaders = [dataloaders, dataloader]
            elif isinstance(dataloaders, list):
                dataloaders.append(dataloader)
            else:
                msg = f"Unknown type for dataloaders {type(dataloaders)}"
                raise TypeError(msg)

        self._setup_dataset_task(dataloaders, datamodule)
        self._setup_transform(model or self.model, datamodule=datamodule, dataloaders=dataloaders, ckpt_path=ckpt_path)
        if self._should_run_validation(model or self.model, None, datamodule, ckpt_path):
            logger.info("Running validation before predicting (threshold / zero-shot setup).")
            self.trainer.validate(
                model,
                dataloaders=None,
                ckpt_path=None,
                verbose=False,
                datamodule=datamodule,
            )

        return self.trainer.predict(model, dataloaders, datamodule, return_predictions, ckpt_path)

    def validate(self, *args, **kwargs) -> _EVALUATE_OUTPUT | None:
        """Intentionally omitted; use ``trainer.validate`` via Anomalib if you need this entrypoint."""
        raise NotImplementedError(
            "`validate` is not implemented on CustomEngine; see Anomalib's Engine for a reference implementation."
        )

    def train(self, *args, **kwargs) -> _EVALUATE_OUTPUT:
        """Intentionally omitted; use ``fit`` which wraps ``trainer.fit`` / validate."""
        raise NotImplementedError(
            "`train` is not implemented on CustomEngine; use `fit` or Anomalib's Engine."
        )

    def export(self, *args, **kwargs) -> Path | None:
        """Intentionally omitted; export via model utilities or Anomalib's Engine."""
        raise NotImplementedError(
            "`export` is not implemented on CustomEngine; see Anomalib's Engine for a reference implementation."
        )
