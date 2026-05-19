"""Data module for the ALLO dataset."""

import logging
from pathlib import Path

import torch
from albumentations import Compose
from pandas import DataFrame

from lightning.pytorch.trainer.states import TrainerFn
from anomalib.data.base import AnomalibDataset
from anomalib.data.utils import (
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    read_image
)

from core.utils import TaskType
from .base import BaseDataModule, FusedTransforms, InputNormalizationMethod
from .utils import read_mask

logger = logging.getLogger(__name__)


def make_allo_dataset(
    split_dir: str | Path,
    split: Split,
) -> DataFrame:
    """Build a sample table for one ALLO split (train or test).

    Args:
        split_dir: Directory containing the split data (``train`` / ``test`` tree).
        split: ``Split.TRAIN`` or ``Split.TEST``.

    Returns:
        DataFrame with columns ``split``, ``image_path``, ``mask_path``, ``label_index``, ``object``.
    """
    columns = ["split", "image_path", "mask_path", "label_index", "object"]
    if split == Split.TRAIN:
        samples = DataFrame([{"split": split.value,
                                "image_path": str(frame),
                                "mask_path": str(frame).replace('images', 'masks'),
                                "label_index": LabelName.NORMAL,
                                "object": None}
                        for frame in split_dir.glob('**/normal/**/images/*.png')],
                    columns=columns)
    else:
        samples = DataFrame([{"split": split.value,
                                "image_path": str(frame),
                                "mask_path": str(frame).replace('images', 'masks'),
                                "label_index": LabelName.ABNORMAL \
                                    if 'anomalous' in str(frame) else LabelName.NORMAL,
                                "object": frame.name.split('_')[1] \
                                    if 'anomalous' in str(frame) else None}
                        for frame in split_dir.glob('**/images/*.png')],
                    columns=columns)
    return samples


class ALLODataset(AnomalibDataset):
    """ALLO segmentation samples built from on-disk folder layout."""

    def __init__(
        self,
        root: Path | str,
        split: Split,
        transform: FusedTransforms,
        sam_transform: FusedTransforms | None = None,
        train_dir: Path | str = "train",
        test_dir: Path | str = "test",
        norm_mask: bool = True,
        anomaly_only_test: bool = False,
        use_anomalies: bool = False,
    ) -> None:
        """Load one split under ``root`` / ``train_dir`` or ``root`` / ``test_dir``.

        Expected layout: ``normal/**/images/*.png`` with parallel ``masks``; test split also
        includes ``anomalous`` branches. Multi-class masks use semantic IDs; when ``norm_mask``
        is True, non-anomaly classes map to 0 and anomaly class 8 maps to 1.
        """
        super().__init__(task=TaskType.SEGMENTATION.value, transform=transform)
        self.root = Path(root)
        self.split = split
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.norm_mask = norm_mask
        self.anomaly_only_test = anomaly_only_test
        self.sam_transform = sam_transform
        self.use_anomalies = use_anomalies
        self._setup()

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Return image, mask (when segmentation), paths, label, and optional ``sam_image``."""
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        image = read_image(image_path, as_tensor=True)
        item = {"image_path": image_path, "label": label_index}

        if self.task == TaskType.SEGMENTATION.value:
            mask = read_mask(mask_path, as_tensor=True).to(torch.uint8)
            assert torch.isin(mask.unique(), torch.tensor([i for i in range(9)] + [255])).all().item(), (
                f"Invalid class in mask {mask_path}"
            )
            if self.split == Split.TRAIN and not self.use_anomalies:
                assert not (mask.unique() == 8).any().item(), (
                    f"Anomaly class (8) in training mask {mask_path}"
                )
            if self.norm_mask:
                mask[(mask < 8) & (mask != 255)] = 0
                mask[(mask == 8) & (mask != 255)] = 1
            item["original_shape"] = torch.tensor(image.shape[-2:])
            item["image"], item["mask"] = self.transform(image, mask) if self.transform else (image, mask)
            if self.sam_transform:
                item["sam_image"] = self.sam_transform(image, mask)[0]
        else:
            msg = f"Unsupported task type: {self.task}"
            raise ValueError(msg)

        return item
    
    def _setup(self) -> None:
        """Populate ``self.samples`` from ``make_allo_dataset`` for this split."""
        if self.split == Split.TRAIN:
            path = self.root / self.train_dir
        else:
            path = self.root / self.test_dir
        self.samples = make_allo_dataset(split_dir=path, split=self.split)
        if self.split == Split.TEST and self.anomaly_only_test:
            self.samples = self.samples[self.samples.label_index == LabelName.ABNORMAL]


class ALLO(BaseDataModule):
    """Lightning datamodule for ALLO train/test with optional same-split override."""

    def __init__(
        self,
        root: Path | str = "./data/ALLO",
        train_dir: Path | str = "train_v3",
        test_dir: Path | str = "test_v3",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        inference_batch_size: int = 32,
        num_workers: int = 8,
        image_size: tuple[int, int] | None = None,
        train_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.3,
        normalization: InputNormalizationMethod = InputNormalizationMethod.ALLO,
        hist_eq: bool = False,
        pad_value: float = 0.0,
        divisor: int | None = None,
        norm_mask_train: bool = True,
        norm_mask_test: bool = True,
        same_train_test: bool = False,
        anomaly_only_test: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            image_size=image_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            normalization=normalization,
            hist_eq=hist_eq,
            pad_value=pad_value,
            divisor=divisor,
            seed=seed,
            **kwargs
        )
        self.root = Path(root)
        self.inference_batch_size = inference_batch_size
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.norm_mask_train = norm_mask_train
        self.norm_mask_test = norm_mask_test
        self.same_train_test = same_train_test
        self.anomaly_only_test = anomaly_only_test

    def _create_test_split(self) -> None:
        """Validate ``test_split_mode`` and warn if the test split has no normal samples."""
        if self.test_split_mode not in [e.value for e in TestSplitMode]:
            msg = f"Unsupported Test Split Mode: {self.test_split_mode}"
            raise ValueError(msg)
        if not self.test_data.has_normal:
            logger.warning("No normal test images found.")

    def _setup(self, _stage: str | None = None) -> None:
        """Build train and test :class:`ALLODataset` instances (``_stage`` is unused)."""
        del _stage

        self.train_data = ALLODataset(
            root=self.root,
            split=Split.TRAIN,
            transform=self.train_transform,
            train_dir=self.train_dir,
            norm_mask=self.norm_mask_train,
            use_anomalies=self.same_train_test
        )
        self.test_data = ALLODataset(
            root=self.root,
            split=Split.TEST,
            transform=self.eval_transform,
            test_dir=self.test_dir,
            norm_mask=self.norm_mask_test,
            anomaly_only_test=self.anomaly_only_test,
            use_anomalies=self.same_train_test
        )

    def setup(self, stage: str | None = None) -> None:
        if not self._is_setup:
            self._setup(stage)
            self._create_test_val_split()
            if isinstance(stage, TrainerFn):
                self._is_setup = True

        if self.same_train_test:
            self.train_data.samples = self.test_data.samples
            logger.warning("Using test data for training. Make sure this is intended!!!")
