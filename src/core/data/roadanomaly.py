"""Data module for RoadAnomaly dataset."""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from albumentations import Compose
from anomalib.data.utils import read_image

from .base import BaseDataModule, FusedTransforms, InputNormalizationMethod
from .utils import read_mask


logger = logging.getLogger(__name__)


class RoadAnomalyDataset(Dataset):
    """RoadAnomaly frames with semantic labels remapped for binary anomaly segmentation."""

    def __init__(self, 
                 root: Path | str, 
                 transform: FusedTransforms,
                 sam_transform: FusedTransforms | None = None,
                 clip_transform: FusedTransforms | None = None) -> None:
        self.transform = transform
        self.sam_transform = sam_transform
        self.clip_transform = clip_transform

        images_dir = Path(root) / "frames"
        self.images = [str(i) for i in images_dir.glob("*.jpg")]
        self.targets = [str(Path(str(i).replace(".jpg", ".labels")) / "labels_semantic.png") for i in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Return tensors with label value 2 remapped to anomaly class 1."""
        image_path = self.images[index]
        mask_path = self.targets[index]
        item = {"image_path": image_path, "target_path": mask_path, "label": 1}

        image, mask = read_image(image_path, as_tensor=True), read_mask(mask_path, as_tensor=True)
        mask[mask == 2] = 1
        allowed = torch.tensor([0, 1, 255])
        assert torch.isin(mask.unique(), allowed).all().item(), f"Invalid Road Anomaly mask values in {mask_path}"

        item["image"], item["mask"] = self.transform(image, mask) if self.transform else (image, mask)
        if self.sam_transform:
            item["sam_image"] = self.sam_transform(image, mask)[0]
        if self.clip_transform:
            item["clip_image"] = self.clip_transform(image, mask)[0]
        item["original_shape"] = torch.tensor(image.shape[-2:])
        return item


class RoadAnomaly(BaseDataModule):
    """Data module for RoadAnomaly dataset."""
    def __init__(
        self,
        root: Path | str = "roadanomaly",
        eval_batch_size: int = 16,
        inference_batch_size: int = 16,
        num_workers: int = 8,
        image_size: tuple[int, int] = (720, 1280),
        eval_transform: Compose | None = None,
        normalization: InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        pad_value: float = 0.0,
        divisor: int | None = None,
        seed: int | None = None,
        **kwargs
    ) -> None:
        super().__init__(
            eval_batch_size=eval_batch_size,
            eval_transform=eval_transform,
            num_workers=num_workers,
            image_size=image_size,
            normalization=normalization,
            pad_value=pad_value,
            divisor=divisor,
            seed=seed,
            **kwargs
        )
        self.root = Path(root)
        self.inference_batch_size = inference_batch_size

    def _setup(self, _stage: str | None = None) -> None:
        del _stage  # Unused variable

        self.val_data = RoadAnomalyDataset(
            root=self.root,
            transform=self.eval_transform
        )
        self.test_data = self.val_data

    def setup(self, stage: str | None = None) -> None:
        self._setup(stage)