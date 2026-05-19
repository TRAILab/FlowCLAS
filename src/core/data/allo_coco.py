"""ALLO training with random COCO OOD regions mixed into each sample."""

import logging
from pathlib import Path

import torch
from albumentations import Compose, Resize

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
from .coco import COCO
from .allo import ALLODataset, make_allo_dataset

logger = logging.getLogger(__name__)


class ALLOCOCODataset(AnomalibDataset):
    """Train split: ALLO in-distribution image plus COCO ``train_id_out`` region pasted for OOD."""

    def __init__(
        self,
        split: Split,
        allo_split_dir: Path | str,
        coco_root: Path | str,
        allo_transform: FusedTransforms,
        coco_mix_transform: FusedTransforms,
        coco_ood_transform: FusedTransforms,
        norm_mask: bool = True,
        ignore_index: int = 255,
        label_start_index: int = 0,
        ignore_coco_bg: bool = False,
        filter_coco: bool = True,
    ) -> None:
        """Wire ALLO sample table, COCO index pool, and three :class:`FusedTransforms` pipelines."""
        super().__init__(task=TaskType.SEGMENTATION.value, transform=None)

        self.split = split
        self.allo_transform = allo_transform
        self.coco_mix_transform = coco_mix_transform
        self.coco_ood_transform = coco_ood_transform
        self.ignore_index = ignore_index
        self.label_start_index = label_start_index
        self.ignore_coco_bg = ignore_coco_bg

        self.samples = make_allo_dataset(
            split_dir=allo_split_dir,
            split=split
        )
        self.norm_mask = norm_mask
        assert not self.has_anomalous, "ALLO train dataset contains anomalous samples."

        self.coco = COCO(coco_root, filter="allo" if filter_coco else None)

    def mix_allo_coco(
        self,
        allo_img: torch.Tensor,
        allo_tgt: torch.Tensor,
        coco_img: torch.Tensor,
        coco_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paste COCO outlier pixels (``train_id_out``) into a random window of the ALLO image and mask."""
        train_id_out = COCO.train_id_out
        coco_tgt[coco_tgt == train_id_out] = train_id_out
        mask = coco_tgt == train_id_out
        box = self.extract_bboxes(mask)
        y1, x1, y2, x2 = box

        coco_tgt_cut = coco_tgt[y1:y2, x1:x2]
        coco_img_cut = coco_img[:, y1:y2, x1:x2]

        h_start = torch.randint(0, allo_tgt.shape[0] - coco_tgt_cut.shape[0] + 1, (1,)).item()
        h_end = h_start + coco_tgt_cut.shape[0]
        w_start = torch.randint(0, allo_tgt.shape[1] - coco_tgt_cut.shape[1] + 1, (1,)).item()
        w_end = w_start + coco_tgt_cut.shape[1]

        allo_img[:, h_start:h_end, w_start:w_end][:, coco_tgt_cut == train_id_out] = (
            coco_img_cut[:, coco_tgt_cut == train_id_out]
        )
        allo_tgt[h_start:h_end, w_start:w_end][coco_tgt_cut == train_id_out] = train_id_out

        return allo_img, allo_tgt

    @staticmethod
    def extract_bboxes(mask: torch.Tensor) -> torch.Tensor:
        """Tight axis-aligned box around True/positive mask pixels ``(y1, x1, y2, x2)`` inclusive-exclusive."""
        if mask.sum() == 0:
            return torch.zeros(4, dtype=torch.int32)

        hori_indices = torch.where(torch.any(mask, dim=0))[0]
        vert_indices = torch.where(torch.any(mask, dim=1))[0]
        x1, x2 = hori_indices[[0, -1]]
        y1, y2 = vert_indices[[0, -1]]
        return torch.tensor([y1, x1, y2+1, x2+1], dtype=torch.int32)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """One mixed train sample: ID ALLO, OOD COCO crop, and pasted mix; all tensors already transformed."""
        allo_img_pth = self.samples.iloc[index].image_path
        allo_tgt_pth = self.samples.iloc[index].mask_path
        coco_idx = self.coco.get_idx()
        coco_img_pth = self.coco.images[coco_idx]
        coco_tgt_pth = self.coco.targets[coco_idx]

        allo_img = read_image(allo_img_pth, as_tensor=True)
        if self.task == TaskType.SEGMENTATION.value:
            allo_tgt = read_mask(allo_tgt_pth, as_tensor=True)
        else:
            msg = f"Unsupported task type: {self.task}"
            raise ValueError(msg)

        coco_img, coco_tgt = read_image(coco_img_pth, as_tensor=True), read_mask(coco_tgt_pth, as_tensor=True)
        if self.ignore_coco_bg:
            coco_tgt[coco_tgt != self.coco.train_id_out] = self.ignore_index

        allo_img, allo_tgt = self.allo_transform(allo_img, allo_tgt)
        coco_ood_img, coco_ood_tgt = self.coco_ood_transform(coco_img.clone(), coco_tgt.clone())
        coco_mix_img, coco_mix_tgt = self.coco_mix_transform(coco_img.clone(), coco_tgt.clone())

        allo_mix_img, allo_mix_tgt = self.mix_allo_coco(
            allo_img.clone(), allo_tgt.clone(), coco_mix_img, coco_mix_tgt
        )

        return {
            "id_image_path": allo_img_pth,
            "ood_image_path": coco_img_pth,
            "id_image": allo_img,
            "id_mask": allo_tgt,
            "ood_image": coco_ood_img,
            "ood_mask": coco_ood_tgt,
            "mix_image": allo_mix_img,
            "mix_mask": allo_mix_tgt,
            "original_shape": torch.tensor(allo_img.shape[-2:]),
            "label": LabelName.ABNORMAL
        }


class ALLOCOCO(BaseDataModule):
    """ALLO + COCO mix for training; ALLO-only :class:`ALLODataset` for test."""

    def __init__(
        self,
        allo_root: Path | str = "./data/ALLO",
        train_dir: Path | str = "train_v3",
        test_dir: Path | str = "test_v3",
        coco_root: Path | str = "./data/coco",
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        inference_batch_size: int = 8,
        num_workers: int = 8,
        image_size: tuple[int, int] | None = None,
        allo_train_transform: Compose | None = None,
        coco_train_mix_transform: Compose | None = None,
        coco_train_ood_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.3,
        normalization: InputNormalizationMethod = InputNormalizationMethod.ALLO,
        norm_mask_train: bool = True,
        norm_mask_test: bool = True,
        label_start_index: int = 0,
        pad_value: float = 0.0,
        mask_pad_value: float = 255,
        divisor: int | None = None,
        different_normalization: bool = False,
        ignore_coco_bg: bool = False,
        filter_coco: bool = False,
        seed: int | None = None,
        anomaly_only_test: bool = False,
        **kwargs
    ) -> None:
        """See parameter descriptions on :class:`BaseDataModule` plus ALLO/COCO-specific fields below.

        Notes:
            Set ``norm_mask_train`` True for ID vs OOD segmentation; False when training full
            semantic classes on ALLO. Set ``norm_mask_test`` False when evaluating semantic IDs.
        """
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            train_transform=None,
            eval_transform=eval_transform,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            image_size=image_size,
            normalization=normalization,
            pad_value=pad_value,
            divisor=divisor,
            seed=seed,
            **kwargs
        )
        self.allo_root = Path(allo_root)
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.coco_root = coco_root
        
        transform_kwargs = dict(
            image_size=image_size,
            normalization=normalization,
            normalize=True,
            hist_eq=False,
            pad_value=pad_value,
            mask_pad_value=mask_pad_value,
            divisor=divisor
        )
        transform_kwargs["image_size"] = None
        self.allo_transform = self._build_transforms(
            transforms=allo_train_transform,
            **transform_kwargs
        )
        transform_kwargs["normalization"] = (
            InputNormalizationMethod.IMAGENET
            if different_normalization
            else normalization)
        self.coco_mix_transform = self._build_transforms(
            transforms=coco_train_mix_transform,
            **transform_kwargs
        )
        self.coco_ood_transform = self._build_transforms(
            transforms=coco_train_ood_transform,
            **transform_kwargs
        )
        transform_kwargs["divisor"] = None
        transform_kwargs["normalize"] = True
        transform_kwargs["normalization"] = InputNormalizationMethod.IMAGENET
        sam_transform = Compose([Resize(height=1024, width=1024, always_apply=True)])
        self.sam_transform = self._build_transforms(
            transforms=sam_transform,
            **transform_kwargs
        )
        self.inference_batch_size = inference_batch_size
        self.mask_pad_value = mask_pad_value
        self.norm_mask_train = norm_mask_train
        self.norm_mask_test = norm_mask_test
        self.label_start_index = label_start_index
        self.ignore_coco_bg = ignore_coco_bg
        self.filter_coco = filter_coco
        self.anomaly_only_test = anomaly_only_test

        self._is_setup = False

    def _setup(self, _stage: str | None = None) -> None:
        del _stage  # Unused

        self.train_data = ALLOCOCODataset(
            split=Split.TRAIN,
            allo_split_dir=self.allo_root / self.train_dir,
            coco_root=self.coco_root,
            allo_transform=self.allo_transform,
            coco_mix_transform=self.coco_mix_transform,
            coco_ood_transform=self.coco_ood_transform,
            norm_mask=self.norm_mask_train,
            ignore_index=self.mask_pad_value,
            label_start_index=self.label_start_index,
            ignore_coco_bg=self.ignore_coco_bg,
            filter_coco=self.filter_coco
        )
        self.test_data = ALLODataset(
            root=self.allo_root,
            split=Split.TEST,
            transform=self.eval_transform,
            sam_transform=self.sam_transform,
            test_dir=self.test_dir,
            norm_mask=self.norm_mask_test,
            anomaly_only_test=self.anomaly_only_test,
        )

    def setup(self, stage: str | None = None) -> None:
        if not self._is_setup:
            self._setup(stage)
            self._create_test_val_split()
            if isinstance(stage, TrainerFn):
                self._is_setup = True
