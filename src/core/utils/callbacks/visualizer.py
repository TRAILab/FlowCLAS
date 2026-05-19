"""Save or log validation/test/predict visualizations."""

import logging
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.data.utils.image import save_image, show_image
from anomalib.loggers import AnomalibWandbLogger
from anomalib.loggers.base import ImageLoggerBase
from anomalib.utils.visualization import (
    BaseVisualizer,
    GeneratorResult,
    MetricsVisualizer,
    VisualizationStep,
)

from core.models import BaseModule
from core.post_processing import ImageVisualizer
from core.utils import TaskType

logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Run image and metric visualizers at batch or stage end."""

    def __init__(
        self,
        visualizers: BaseVisualizer | list[BaseVisualizer] | None = None,
        save: bool = False,
        root: Path | str = "images",
        log: bool = False,
        show: bool = False,
        total: int | None = None,
    ) -> None:
        if visualizers is None:
            visualizers = []
        if save and root is None:
            raise ValueError("`root` is required when `save` is True.")
        self.save = save
        self.root: Path = Path(root) if root is not None else Path()
        self.log = log
        self.show = show
        self.generators = visualizers if isinstance(visualizers, list) else [visualizers]
        self.cnt = 0
        self.total = total

    def on_test_end(self, trainer: Trainer, pl_module: BaseModule) -> None:
        for generator in self.generators:
            if (
                generator.visualize_on == VisualizationStep.STAGE_END
                and isinstance(generator, MetricsVisualizer)
            ):
                for result in generator(trainer=trainer, pl_module=pl_module):
                    if self.save:
                        if result.file_name is None:
                            raise ValueError("Cannot save visualization without a file name.")
                        save_image(
                            image=result.image,
                            root=self.root / "metrics",
                            filename=result.file_name,
                        )
                    if self.show:
                        show_image(image=result.image, title=str(result.file_name))
                    if self.log:
                        self._add_to_logger(result, pl_module, trainer)

        if self.log:
            for logger in trainer.loggers:
                if isinstance(logger, AnomalibWandbLogger):
                    logger.save()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.total is not None and self.cnt >= self.total:
            return

        for generator in self.generators:
            if generator.visualize_on != VisualizationStep.BATCH:
                continue
            if not isinstance(generator, ImageVisualizer):
                continue

            for image, _mask, heatmap, histogram in generator(
                trainer=trainer,
                pl_module=pl_module,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            ):
                if not self.save:
                    continue
                save_image(
                    image=image.image,
                    root=self.root / "images",
                    filename=image.file_name.name,
                )
                save_image(
                    image=heatmap.image,
                    root=self.root / "images",
                    filename=heatmap.file_name.name,
                )
                save_image(
                    image=histogram.image,
                    root=self.root / "images",
                    filename=histogram.file_name.name,
                )
                self.cnt += 1

    def _add_to_logger(
        self,
        result: GeneratorResult,
        module: BaseModule,
        trainer: Trainer,
    ) -> None:
        if result.file_name is None:
            raise ValueError("File name is required for logging.")

        available_loggers = {
            type(logger).__name__.lower().replace("logger", "").replace("anomalib", ""): logger
            for logger in trainer.loggers
        }
        filename = result.file_name
        for logger in available_loggers.values():
            if isinstance(logger, ImageLoggerBase):
                image_logger = cast(ImageLoggerBase, logger)
                name = (
                    f"{filename.parent.name}_{filename.name}"
                    if isinstance(filename, Path)
                    else filename
                )
                image_logger.add_image(
                    image=result.image,
                    name=name,
                    global_step=module.global_step,
                )


def get_visualizer_callback(
    visualizers: list[str] | None = None,
    save: bool = False,
    root: Path | str = "images",
    total: int | None = None,
    features_only: bool = False,
    task: TaskType = TaskType.SEGMENTATION,
) -> VisualizationCallback:
    """Build the default image + metrics visualizer callback for the engine."""
    del visualizers  # CLI may pass names; engine always uses the defaults below.
    return VisualizationCallback(
        visualizers=[
            ImageVisualizer(task=task, features_only=features_only),
            MetricsVisualizer(),
        ],
        save=save,
        root=root,
        total=total,
    )
