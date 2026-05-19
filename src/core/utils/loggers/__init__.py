"""Factory for PyTorch Lightning experiment loggers (Anomalib + CSV)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from jsonargparse import Namespace
from lightning.pytorch.loggers import CSVLogger, Logger

from anomalib.loggers.comet import AnomalibCometLogger
from anomalib.loggers.tensorboard import AnomalibTensorBoardLogger

from .wandb import CustomWandbLogger

__all__ = ["get_experiment_logger"]

AVAILABLE_LOGGERS = ("tensorboard", "wandb", "csv", "comet")


class UnknownLogger(ValueError):
    """Raised when ``experiment.logger`` contains an unsupported backend name."""


def get_experiment_logger(
    config: Namespace | None,
    project_dir: str | Path,
) -> list[Logger]:
    """Build Lightning loggers from ``config.logger`` (string or list of names).

    Args:
        config: Experiment namespace with ``logger``, ``log_graph``, ``project_name``,
            ``name``, ``offline``, and ``wandb_id`` as required by each backend.
        project_dir: Run root; logs are written under ``{project_dir}/logs``.

    Returns:
        One logger instance per configured backend, in config order.

    Raises:
        UnknownLogger: If any name is not in ``AVAILABLE_LOGGERS``.
    """
    if config is None:
        return []

    logger_list: list[Logger] = []
    log_dir = Path(project_dir) / "logs"
    backends = config.logger if isinstance(config.logger, list) else [config.logger]

    for backend in backends:
        if backend == "tensorboard":
            logger_list.append(
                AnomalibTensorBoardLogger(
                    name="Tensorboard Logs",
                    save_dir=log_dir,
                    log_graph=config.log_graph,
                )
            )
        elif backend == "wandb":
            log_dir.mkdir(parents=True, exist_ok=True)
            logger_list.append(
                CustomWandbLogger(
                    project=config.project_name,
                    name=config.name,
                    save_dir=log_dir,
                    offline=config.offline,
                    id=config.wandb_id,
                )
            )
        elif backend == "comet":
            log_dir.mkdir(parents=True, exist_ok=True)
            logger_list.append(
                AnomalibCometLogger(
                    project_name=config.project_name,
                    experiment_name=config.name,
                    save_dir=log_dir,
                )
            )
        elif backend == "csv":
            logger_list.append(CSVLogger(save_dir=log_dir))
        else:
            raise UnknownLogger(
                f"Unknown logger {backend!r}; choose from {list(AVAILABLE_LOGGERS)}."
            )

    return logger_list
