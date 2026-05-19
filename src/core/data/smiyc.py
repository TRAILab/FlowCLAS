"""SMIYC AnomalyTrack and ObstacleTrack validation splits."""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from anomalib.data.utils import read_image

from .base import FusedTransforms
from .utils import read_mask


logger = logging.getLogger(__name__)


class SMIYCDataset(Dataset):
    """Shared logic for SMIYC tracks (image/mask paths filled by subclasses)."""

    def __init__(
        self,
        transform: FusedTransforms,
        sam_transform: FusedTransforms | None = None,
        clip_transform: FusedTransforms | None = None,
    ) -> None:
        self.transform = transform
        self.sam_transform = sam_transform
        self.clip_transform = clip_transform

        self.images: list[str]
        self.targets: list[str]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Return validation tensors plus optional SAM/CLIP views."""
        image_path = self.images[index]
        mask_path = self.targets[index]
        item = {"image_path": image_path, "target_path": mask_path, "label": 1}

        image, mask = read_image(image_path, as_tensor=True), read_mask(mask_path, as_tensor=True)

        item["image"], item["mask"] = self.transform(image, mask) if self.transform else (image, mask)
        if self.sam_transform:
            item["sam_image"] = self.sam_transform(image, mask)[0]
        if self.clip_transform:
            item["clip_image"] = self.clip_transform(image, mask)[0]
        item["original_shape"] = torch.tensor(image.shape[-2:])
        return item


class SMIYC_AnomalyTrack(SMIYCDataset):
    """``dataset_AnomalyTrack`` validation JPEGs."""

    def __init__(
        self,
        root: Path | str,
        transform: FusedTransforms,
        sam_transform: FusedTransforms | None = None,
        clip_transform: FusedTransforms | None = None,
    ) -> None:
        super().__init__(transform, sam_transform, clip_transform)

        anomalytrack_dir = Path(root) / "dataset_AnomalyTrack" / "images"
        self.images = [str(i) for i in anomalytrack_dir.glob("validation*.jpg")]
        self.targets = [
            str(i).replace("images", "labels_masks").replace(".jpg", "_labels_semantic.png")
            for i in self.images
        ]

        assert len(self.images) == len(self.targets), "Number of images and masks should be same."


class SMIYC_ObstacleTrack(SMIYCDataset):
    """``dataset_ObstacleTrack`` validation WebP images."""

    def __init__(
        self,
        root: Path | str,
        transform: FusedTransforms,
        sam_transform: FusedTransforms | None = None,
        clip_transform: FusedTransforms | None = None,
    ) -> None:
        super().__init__(transform, sam_transform, clip_transform)

        obstacletrack_dir = Path(root) / "dataset_ObstacleTrack" / "images"
        self.images = [str(i) for i in obstacletrack_dir.glob("validation*.webp")]
        self.targets = [
            str(i).replace("images", "labels_masks").replace(".webp", "_labels_semantic.png")
            for i in self.images
        ]

        assert len(self.images) == len(self.targets), "Number of images and masks should be same."
