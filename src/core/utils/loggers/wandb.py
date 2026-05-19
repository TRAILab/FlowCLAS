"""Weights & Biases logger with explicit step alignment for Lightning metrics."""

from typing import Mapping, Optional

from anomalib.loggers.wandb import AnomalibWandbLogger
from lightning.fabric.utilities.logger import _add_prefix
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override


class CustomWandbLogger(AnomalibWandbLogger):
    """Anomalib W&B logger that always passes Lightning's ``step`` to ``wandb.log``."""

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            self.experiment.log(
                dict(metrics, **{"trainer/global_step": step}),
                step=step,
            )
        else:
            self.experiment.log(metrics)
