"""FlowCLAS CLI. Adapted from Anomalib CLI."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from rich import traceback
from torch.utils.data import DataLoader, Dataset

try:
    from lightning.pytorch import Trainer, seed_everything
except ImportError as exc:
    raise ImportError("PyTorch Lightning is not installed.") from exc

from anomalib import __version__
from anomalib.cli.utils.help_formatter import CustomHelpFormatter, get_short_docstring
from anomalib.loggers import configure_logger
from anomalib.metrics.threshold import BaseThreshold

import core.utils.schedulers  # noqa: F401 — registers schedulers for jsonargparse class_path resolution
from core.data import BaseDataModule
from core.engine import CustomEngine as Engine
from core.models import BaseModule
from core.utils import TaskType
from core.utils.metrics import BaseMetric

# Optional: clone location when not installed as a package (set SAM2_REPO to repo root, e.g. /sam2).
_sam2_repo = os.environ.get("SAM2_REPO")
if _sam2_repo and (_root := Path(_sam2_repo)).is_dir():
    sys.path.insert(0, str(_root))

traceback.install()
logger = logging.getLogger("FlowCLAS.cli")


class FlowCLASCLI:
    """Implementation of a fully configurable CLI tool.

    The advantage of this tool is its flexibility to configure the pipeline
    from both the CLI and a configuration file (.yaml or .json). It is even
    possible to use both the CLI and a configuration file simultaneously.
    For more details, the reader could refer to PyTorch Lightning CLI
    documentation.

    ``save_config_kwargs`` is set to ``overwrite=True`` so that the
    ``SaveConfigCallback`` overwrites the config if it already exists.
    """

    def __init__(self, args: Sequence[str] | None = None, run: bool = False) -> None:
        self.parser = self.init_parser()
        self.subcommand_parsers: dict[str, ArgumentParser] = {}
        self.subcommand_method_arguments: dict[str, list[str]] = {}
        self.add_subcommands()
        self.config = self.parser.parse_args(args=args)
        self.subcommand = self.config["subcommand"]
        self.before_instantiate_classes()
        self.instantiate_classes()
        if run:
            self._run_subcommand()

    def init_parser(self, **kwargs) -> ArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault("dump_header", [f"anomalib=={__version__}"])
        parser = ArgumentParser(formatter_class=CustomHelpFormatter, **kwargs)
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        return parser

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Skip predict subcommand as it is added later."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
        }

    @staticmethod
    def custom_subcommands() -> dict[str, dict[str, str]]:
        """Return a dictionary of subcommands and their description."""
        return {
            "predict": {"description": "Run inference on a model."},
        }

    def add_subcommands(self, **kwargs) -> None:
        """Initialize base subcommands and add custom subcommands on top of it."""
        parser_subcommands = self.parser.add_subcommands()

        # Add Trainer subcommands
        for subcommand in self.subcommands():
            sub_parser = self.init_parser(**kwargs)

            fn = getattr(Trainer, subcommand)
            # First line of docstring becomes the subcommand help text
            description = get_short_docstring(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault("description", description)

            self.subcommand_parsers[subcommand] = sub_parser
            parser_subcommands.add_subcommand(subcommand, sub_parser, help=description)
            self.add_trainer_arguments(sub_parser, subcommand)

        # Add custom subcommands
        custom = self.custom_subcommands()
        for subcommand, meta in custom.items():
            sub_parser = self.init_parser(**kwargs)

            self.subcommand_parsers[subcommand] = sub_parser
            parser_subcommands.add_subcommand(
                subcommand,
                sub_parser,
                help=meta["description"],
            )
            getattr(self, f"add_{subcommand}_arguments")(sub_parser)

    def add_experiment_arguments_to_parser(self, parser: ArgumentParser) -> None:
        """Add experiment arguments to the parser."""
        parser.add_argument(
            "--experiment.logger",
            type=str | list[str],
            required=False,
            default="csv",
            help="Logger(s) to use for logging.",
        )
        parser.add_argument(
            "--experiment.log_graph",
            type=bool,
            required=False,
            default=False,
            help="Whether to log the computation graph.",
        )
        parser.add_argument(
            "--experiment.name",
            type=str,
            required=True,
            help="Name of experiment run.",
        )
        parser.add_argument(
            "--experiment.project_name",
            type=str,
            required=True,
            help="Name of project group for this experiment run.",
        )
        parser.add_argument(
            "--experiment.time_delta",
            type=int,
            required=False,
            default=60,
            help="Seconds since last directory creation that determines whether to create a new version.",
        )
        parser.add_argument(
            "--experiment.offline",
            type=bool,
            required=False,
            default=False,
            help="Whether to run the experiment in offline mode."
        )
        parser.add_argument(
            "--experiment.wandb_id",
            type=str,
            required=False,
            default=None,
            help="Wandb ID for the experiment."
        )

    def add_engine_arguments_to_parser(self, parser: ArgumentParser) -> None:
        """Extend trainer's arguments to add engine arguments.

        .. note::
            Since ``Engine`` parameters are manually added, any change to the
            ``Engine`` class should be reflected manually.
        """
        from core.utils.callbacks import get_visualizer_callback

        parser.add_function_arguments(get_visualizer_callback, "visualization")
        parser.add_argument("--task", type=TaskType | str, default=TaskType.SEGMENTATION)
        parser.add_argument("--metrics_on_cpu", action="store_true", default=False)
        parser.add_argument(
            "--metrics.image",
            type=list[BaseMetric] | list[str] | str | None,
            default=[],
        )
        parser.add_argument(
            "--metrics.pixel",
            type=list[BaseMetric] | list[str] | str | None,
            default=[],
        )
        parser.add_argument("--metrics.threshold", type=BaseThreshold | str, default="F1Threshold")
        parser.add_argument("--monitor1.metric", type=str | list[str], required=True)
        parser.add_argument("--monitor1.mode", type=str, required=True, choices=["min", "max"])
        parser.add_argument("--monitor2.metric", type=str | list[str], required=False)
        parser.add_argument("--monitor2.mode", type=str, required=False, choices=["min", "max"])
        if getattr(parser, "subcommand", None) != "predict":
            parser.link_arguments("task", "data.init_args.task")
        parser.add_argument(
            "--work_dir",
            type=Path,
            help="Path to save the results.",
            default=Path("../results"),
        )
        parser.link_arguments("work_dir", "trainer.default_root_dir")
        parser.add_argument("--save_every_epoch", action="store_true")
        parser.add_argument("--no_sigmoid", action="store_false", dest="apply_sigmoid")
        parser.add_argument("--ema", action="store_true", help="Whether to use EMA.")

        self.add_experiment_arguments_to_parser(parser)

    def add_trainer_arguments(self, parser: ArgumentParser, subcommand: str) -> None:
        """Add train arguments to the parser."""
        self._add_default_arguments_to_parser(parser)
        self._add_trainer_arguments_to_parser(parser, add_optimizer=True, add_scheduler=True)
        parser.add_subclass_arguments(
            BaseModule,
            "model",
            fail_untyped=False,
            required=True,
        )
        parser.add_subclass_arguments(BaseDataModule, "data")
        self.add_engine_arguments_to_parser(parser)
        skip: set[str | int] = set(self.subcommands()[subcommand])
        added = parser.add_method_arguments(
            Trainer,
            subcommand,
            skip=skip,
        )
        self.subcommand_method_arguments[subcommand] = added

    def add_predict_arguments(self, parser: ArgumentParser) -> None:
        """Add predict arguments to the parser."""
        self._add_default_arguments_to_parser(parser)
        self._add_trainer_arguments_to_parser(parser)
        parser.add_subclass_arguments(
            BaseModule,
            "model",
            fail_untyped=False,
            required=True,
        )
        parser.add_subclass_arguments(BaseDataModule, "data")
        added = parser.add_method_arguments(
            Engine,
            "predict",
            skip={"model", "dataloaders", "datamodule", "dataset"},
        )
        self.subcommand_method_arguments["predict"] = added
        self.add_engine_arguments_to_parser(parser)

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes and sets up tiler."""
        subcommand = self.config["subcommand"]
        if subcommand in (*self.subcommands(), "predict"):
            if subcommand == "predict" and isinstance(self.config["predict"]["data"], str | Path):
                raise NotImplementedError("Predicting from a path is not yet supported.")

    def instantiate_classes(self) -> None:
        """Instantiate classes depending on the subcommand.

        For trainer related commands it instantiates all the model, datamodule and trainer classes.
        But for subcommands we do not want to instantiate any trainer specific classes such as datamodule, model, etc
        This is because the subcommand is responsible for instantiating and executing code based on the passed config
        """
        if self.config["subcommand"] in (*self.subcommands(), *self.custom_subcommands()):
            self.config[self.subcommand] = self._set_seeds(self.config[self.subcommand])
            self.config[self.subcommand].data = self._scale_batch_size(self._get(self.config, "data"),
                                                                       int(self._get(self.config, "trainer").devices))
            self.config_init = self.parser.instantiate_classes(self.config)
            self.datamodule = self._get(self.config_init, "data")
            if isinstance(self.datamodule, Dataset):
                self.datamodule = DataLoader(self.datamodule)
            self.model = self._get(self.config_init, "model")
            if self.config["subcommand"] not in self.custom_subcommands():
                self._set_optimizer_scheduler_cfg()
            self.instantiate_engine()
        else:
            raise ValueError(f"Subcommand {self.config['subcommand']} is not supported.")

    def instantiate_engine(self) -> None:
        """Instantiate the engine.

        .. note::
            Most of the code in this method is taken from ``LightningCLI``'s
            ``instantiate_trainer`` method. Refer to that method for more
            details.
        """
        from lightning.pytorch.cli import SaveConfigCallback

        from anomalib.callbacks import get_callbacks

        engine_args = {
            "threshold": self._get(self.config_init, "metrics.threshold"),
            "task": self._get(self.config_init, "task"),
            "metrics_on_cpu": self._get(self.config_init, "metrics_on_cpu"),
            "image_metrics": self._get(self.config_init, "metrics.image"),
            "pixel_metrics": self._get(self.config_init, "metrics.pixel"),
            "experiment_cfg": self._get(self.config_init, "experiment"),
            "visualization": self._get(self.config_init, "visualization"),
            "monitor1": self._get(self.config_init, "monitor1"),
            "monitor2": self._get(self.config_init, "monitor2"),
            "save_every_epoch": self._get(self.config_init, "save_every_epoch"),
            "apply_sigmoid": self._get(self.config_init, "apply_sigmoid", True),
            "ema": self._get(self.config_init, "ema", False),
        }
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **engine_args}
        if int(trainer_config.get("devices", 1)) == 1:
            trainer_config["strategy"] = "auto"
        if "callbacks" in trainer_config:
            if trainer_config["callbacks"] is None:
                trainer_config["callbacks"] = []
            elif not isinstance(trainer_config["callbacks"], list):
                trainer_config["callbacks"] = [trainer_config["callbacks"]]
            if not trainer_config.get("fast_dev_run", False):
                config_callback = SaveConfigCallback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    overwrite=True,
                )
                trainer_config["callbacks"].append(config_callback)
        trainer_config["callbacks"].extend(get_callbacks(self.config[self.subcommand]))
        self.engine = Engine(**trainer_config)

    def _run_subcommand(self) -> None:
        """Run subcommand depending on the subcommand.

        This overrides the original ``_run_subcommand`` to run the ``Engine``
        method rather than the ``Train`` method.
        """
        logger.info(f"Running subcommand: {self.subcommand}")

        if self.config["subcommand"] in (*self.subcommands(), *self.custom_subcommands()):
            fn = getattr(self.engine, self.subcommand)
            fn_kwargs = self._prepare_subcommand_kwargs(self.subcommand)
            fn(**fn_kwargs)
        else:
            raise ValueError(f"Subcommand {self.config['subcommand']} is not supported.")

    @property
    def fit(self) -> Callable:
        """Fit the model using engine's fit method."""
        return self.engine.fit

    @property
    def validate(self) -> Callable:
        """Validate the model using engine's validate method."""
        return self.engine.validate

    @property
    def test(self) -> Callable:
        """Test the model using engine's test method."""
        return self.engine.test

    @property
    def predict(self) -> Callable:
        """Predict using engine's predict method."""
        return self.engine.predict

    def _set_seeds(self, config: Namespace) -> Namespace:
        """Set all seeds to the same value."""
        if isinstance(config.seed_everything, int):
            logger.info(f"Setting seed to {config.seed_everything}")
            seed_everything(config.seed_everything)
            for key in config.keys():
                if "seed" in key and key != "seed_everything":
                    config[key] = config.seed_everything
        else:
            raise ValueError(f"seed_everything is not an int: {config.seed_everything}")
        return config

    def _scale_batch_size(self, config: Namespace, num_devices: int) -> Namespace:
        """Scale the batch size in the config according to the number of devices."""
        if num_devices > 1:
            logger.info(f"Dividing batch size by {num_devices}. This assumes that the batch size in config is for a single device.")
            for key in config.init_args.keys():
                if key in ["train_batch_size", "eval_batch_size", "inference_batch_size"]:
                    batch_size = config.init_args[key]
                    if batch_size % num_devices != 0:
                        raise ValueError(f"Batch size {batch_size} is not divisible by the number of devices {num_devices}.")
                    config.init_args[key] = batch_size // num_devices

        return config

    def _add_trainer_arguments_to_parser(
        self,
        parser: ArgumentParser,
        add_optimizer: bool = False,
        add_scheduler: bool = False,
    ) -> None:
        """Add trainer arguments to the parser and optionally optimizer and scheduler."""
        parser.add_class_arguments(Trainer, "trainer", fail_untyped=False, instantiate=False, sub_configs=True)

        if add_optimizer:
            from torch.optim import Optimizer

            optim_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"params"}}
            parser.add_subclass_arguments(
                baseclass=(Optimizer,),
                nested_key="optimizer",
                **optim_kwargs,
            )
        else:
            parser.add_argument("--optimizer", type=dict, default=None, help="Placeholder (not used).")
        if add_scheduler:
            from torch.optim.lr_scheduler import LRScheduler

            scheduler_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"optimizer"}}
            parser.add_subclass_arguments(
                baseclass=LRScheduler,
                nested_key="lr_scheduler",
                **scheduler_kwargs,
            )
        else:
            parser.add_argument("--lr_scheduler", type=dict, default=None, help="Placeholder (not used).")

    def _add_default_arguments_to_parser(self, parser: ArgumentParser) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument(
            "--seed_everything",
            type=bool | int,
            default=True,
            help=(
                "Set to an int to run seed_everything with this value before classes instantiation."
                "Set to True to use a random seed."
            ),
        )

    def _get(self, config: Namespace, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v for k, v in self.config_init[subcommand].items() if k in self.subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            if isinstance(self.datamodule, BaseDataModule):
                fn_kwargs["datamodule"] = self.datamodule
            elif isinstance(self.datamodule, DataLoader):
                fn_kwargs["dataloaders"] = self.datamodule
        return fn_kwargs

    def _parser(self, subcommand: str | None) -> ArgumentParser:
        if subcommand is None:
            return self.parser
        return self.subcommand_parsers[subcommand]

    def _set_optimizer_scheduler_cfg(self) -> None:
        """Saves optimizer and lr_scheduler config as BaseModule property."""
        optimizer_cfg = self._get(self.config_init, "optimizer", None)
        if optimizer_cfg is None:
            raise ValueError("Optimizer config is not found in the config.")
        lr_scheduler_cfg = self._get(self.config_init, "lr_scheduler", {})
        if isinstance(self.model, BaseModule):
            self.model.set_optimizer_scheduler_cfg(optimizer_cfg, lr_scheduler_cfg)


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    FlowCLASCLI(run=True)


if __name__ == "__main__":
    main()
