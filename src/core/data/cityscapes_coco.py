"""Cityscapes fine semantic labels combined with COCO OOD mixing for training."""

from pathlib import Path

import torch
from torchvision.tv_tensors import Mask
from torchvision.datasets import Cityscapes as _CityscapesDataset
from torch.utils.data.dataloader import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import albumentations as A
from albumentations import Compose

from anomalib.data.utils import read_image, LabelName
from anomalib.data.base.datamodule import collate_fn

from .base import BaseDataModule, FusedTransforms, InputNormalizationMethod
from .utils import read_mask
from .fishyscapes import FishyscapesDataset
from .coco import COCO
from .smiyc import SMIYC_AnomalyTrack, SMIYC_ObstacleTrack
from .roadanomaly import RoadAnomalyDataset

class CityscapesCOCODataset(_CityscapesDataset):
    """One Cityscapes train image paired with a random COCO ``ood_seg`` pair and fused augmentations."""

    CLASS_NAMES = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )
    CLASS_INDICES = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
    VOID_INDICES = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)

    def __init__(
        self,
        coco_root: Path | str,
        city_transform: FusedTransforms,
        coco_mix_transform: FusedTransforms,
        coco_ood_transform: FusedTransforms,
        sam_transform: FusedTransforms,
        clip_transform: FusedTransforms,
        ignore_index: int = 255,
        biased: bool = False,
        filter_coco: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.coco = COCO(coco_root, filter=("road_anomaly" if filter_coco else None))
        self.city_transform = city_transform
        self.coco_mix_transform = coco_mix_transform
        self.coco_ood_transform = coco_ood_transform
        self.sam_transform = sam_transform
        self.clip_transform = clip_transform
        self.ignore_index = ignore_index
        self.biased = biased

    def encode_target_city(self, target: Mask) -> Mask:
        """Remap Cityscapes IDs to contiguous class indices; void classes become ``ignore_index``."""
        for void_index in self.VOID_INDICES:
            target[target == void_index] = self.ignore_index
        for i, cls_index in enumerate(self.CLASS_INDICES):
            target[target == cls_index] = i
        return target

    def mix_city_coco(self, 
                      city_img: torch.Tensor,
                      city_tgt: torch.Tensor,
                      coco_img: torch.Tensor,
                      coco_tgt: torch.Tensor    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paste COCO ``train_id_out`` pixels into a random Cityscapes crop (optional spatial bias)."""
        train_id_out = COCO.train_id_out
        coco_tgt[coco_tgt == train_id_out] = train_id_out
        mask = coco_tgt == train_id_out
        box = self.extract_bboxes(mask)
        y1, x1, y2, x2 = box

        coco_tgt_cut = coco_tgt[y1:y2, x1:x2]
        coco_img_cut = coco_img[:, y1:y2, x1:x2]

        min_h = int(city_tgt.shape[0] * 0.25) if self.biased else 0
        h_start = torch.randint(min_h, city_tgt.shape[0] - coco_tgt_cut.shape[0] + 1, (1,)).item()
        h_end = h_start + coco_tgt_cut.shape[0]
        min_w = int(city_tgt.shape[1] * 0.1) if self.biased else 0
        max_w = int(city_tgt.shape[1] * 0.9) if self.biased else city_tgt.shape[1]
        w_start = torch.randint(min_w, max_w - coco_tgt_cut.shape[1] + 1, (1,)).item()
        w_end = w_start + coco_tgt_cut.shape[1]

        city_img[:, h_start:h_end, w_start:w_end][:, coco_tgt_cut == train_id_out] = (
            coco_img_cut[:, coco_tgt_cut == train_id_out]
        )
        city_tgt[h_start:h_end, w_start:w_end][coco_tgt_cut == train_id_out] = train_id_out

        return city_img, city_tgt

    @staticmethod
    def extract_bboxes(mask: torch.Tensor) -> torch.Tensor:
        """Axis-aligned bounding box ``(y1, x1, y2, x2)`` around positive mask pixels."""
        if mask.sum() == 0:
            return torch.zeros(4, dtype=torch.int32)

        hori_indices = torch.where(torch.any(mask, dim=0))[0]
        vert_indices = torch.where(torch.any(mask, dim=1))[0]
        x1, x2 = hori_indices[[0, -1]]
        y1, y2 = vert_indices[[0, -1]]
        return torch.tensor([y1, x1, y2+1, x2+1], dtype=torch.int32)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Return ID Cityscapes, OOD COCO, SAM/CLIP views of the mix, and normalized tensors."""
        city_img_pth = self.images[index]
        if self.target_type[0] != "semantic":
            raise ValueError(f"Target type {self.target_type} is not supported for this Cityscapes dataset class.")
        city_tgt_pth = self.targets[index][0]
        coco_idx = self.coco.get_idx()
        coco_img_pth = self.coco.images[coco_idx]
        coco_tgt_pth = self.coco.targets[coco_idx]

        city_img, city_tgt = read_image(city_img_pth, as_tensor=True), \
                                self.encode_target_city(read_mask(city_tgt_pth, as_tensor=True))
        coco_img, coco_tgt = read_image(coco_img_pth, as_tensor=True), read_mask(coco_tgt_pth, as_tensor=True)

        city_img, city_tgt = self.city_transform(city_img, city_tgt)
        coco_ood_img, coco_ood_tgt = self.coco_ood_transform(coco_img.clone(), coco_tgt.clone())
        coco_mix_img, coco_mix_tgt = self.coco_mix_transform(coco_img.clone(), coco_tgt.clone())

        city_mix_img, city_mix_tgt = self.mix_city_coco(
            city_img.clone(), city_tgt.clone(), coco_mix_img, coco_mix_tgt
        )

        mix_img_unnorm = city_mix_img.clone()

        city_mix_img_sam = self.sam_transform(city_mix_img.clone(), city_mix_tgt.clone())[0]
        city_mix_img_clip = self.clip_transform(city_mix_img.clone(), city_mix_tgt.clone())[0]

        city_img, city_tgt = self.city_transform.normalize_transform(city_img, city_tgt)
        city_mix_img = self.city_transform.normalize_transform(city_mix_img)
        coco_ood_img = self.coco_ood_transform.normalize_transform(coco_ood_img)

        return {
            "id_image_path": city_img_pth,
            "ood_image_path": coco_img_pth,
            "id_image": city_img,
            "id_mask": city_tgt,
            "ood_image": coco_ood_img,
            "ood_mask": coco_ood_tgt,
            "mix_image": city_mix_img,
            "mix_image_unnorm": mix_img_unnorm,
            "mix_image_sam": city_mix_img_sam,
            "mix_image_clip": city_mix_img_clip,
            "mix_mask": city_mix_tgt,
            "original_shape": torch.tensor(city_img.shape[-2:]),
            "label": LabelName.ABNORMAL
        }

    def __len__(self) -> int:
        """Number of Cityscapes images in this split."""
        return len(self.images)


class CityscapesCOCO(BaseDataModule):
    """Train on Cityscapes+COCO mix; evaluate on Fishyscapes, SMIYC tracks, and RoadAnomaly."""

    def __init__(
        self,
        city_root: Path | str = "./data/cityscapes",
        coco_root: Path | str = "./data/coco",
        fishy_root: Path | str = "./data/fishyscapes",
        smiyc_root: Path | str = "./data/AnomalyTrack",
        roadanomaly_root: Path | str = "./data/RoadAnomaly",
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        inference_batch_size: int = 8,
        num_workers: int = 8,
        image_size: tuple[int, int] | None = None,
        city_train_transform: Compose | None = None,
        coco_train_mix_transform: Compose | None = None,
        coco_train_ood_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        normalization: InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        pad_value: float = 0.0,
        divisor: int | None = None,
        biased: bool = False,
        seed: int | None = None,
        filter_coco: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            train_transform=None,
            eval_transform=eval_transform,
            num_workers=num_workers,
            image_size=image_size,
            normalization=normalization,
            pad_value=pad_value,
            divisor=divisor,
            seed=seed,
            **kwargs
        )
        self.city_root = city_root
        self.coco_root = coco_root
        self.fishy_root = fishy_root
        self.smiyc_root = smiyc_root
        self.roadanomaly_root = roadanomaly_root
        self.filter_coco = filter_coco

        transform_kwargs = dict(
            image_size=None,
            normalization=normalization,
            normalize=False,
            hist_eq=False,
            pad_value=pad_value,
            mask_pad_value=255,
            divisor=divisor
        )
        self.city_transform = self._build_transforms(
            transforms=city_train_transform,
            **transform_kwargs
        )
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
        sam_transform = A.Compose([A.Resize(height=1024, width=1024, always_apply=True)])
        self.sam_transform = self._build_transforms(
            transforms=sam_transform,
            **transform_kwargs
        )

        transform_kwargs["normalization"] = InputNormalizationMethod.CLIP
        transform_kwargs["divisor"] = 14
        self.clip_transform = self._build_transforms(
            transforms=None,
            **transform_kwargs
        )
        self.inference_batch_size = inference_batch_size
        self.biased = biased

    def _setup(self, _stage: str | None = None) -> None:
        del _stage  # Unused

        self.train_data = CityscapesCOCODataset(
            root=self.city_root,
            split="train",
            mode="fine",
            target_type="semantic",
            coco_root=self.coco_root,
            city_transform=self.city_transform,
            coco_mix_transform=self.coco_mix_transform,
            coco_ood_transform=self.coco_ood_transform,
            sam_transform=self.sam_transform,
            clip_transform=self.clip_transform,
            biased=self.biased,
            filter_coco=self.filter_coco
        )

        self.fishyscapes = FishyscapesDataset(root=self.fishy_root, 
                                              transform=self.eval_transform,
                                              sam_transform=self.sam_transform,
                                              clip_transform=self.clip_transform)
        self.anomalytrack = SMIYC_AnomalyTrack(root=self.smiyc_root, 
                                                transform=self.eval_transform,
                                                sam_transform=self.sam_transform,
                                                clip_transform=self.clip_transform)
        self.obstacletrack = SMIYC_ObstacleTrack(root=self.smiyc_root, 
                                                transform=self.eval_transform,
                                                sam_transform=self.sam_transform,
                                                clip_transform=self.clip_transform)
        self.roadanomaly = RoadAnomalyDataset(
            root=self.roadanomaly_root,
            transform=self.eval_transform,
            sam_transform=self.sam_transform,
            clip_transform=self.clip_transform,
        )
        benchmark_sets = [
            self.fishyscapes,
            self.anomalytrack,
            self.obstacletrack,
            self.roadanomaly,
        ]
        benchmark_names = ["Fishy", "AnomalyTrack", "ObstacleTrack", "RoadAnomaly"]
        active = (0, 1, 2, 3)
        self.test_data_list = [benchmark_sets[i] for i in active]
        self.data_names = [benchmark_names[i] for i in active]

    def setup(self, stage: str | None = None) -> None:
        self._setup(stage)

    def _get_val_loader(self, data) -> DataLoader:
        """Return a :class:`DataLoader` for one benchmark ``Dataset``."""
        return DataLoader(
            dataset=data,
            shuffle=self.shuffle,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers if not self.debug else 0,
            collate_fn=collate_fn,
            pin_memory=(not self.debug),
            persistent_workers=(not self.debug),
        )

    def _get_test_loader(self, data) -> DataLoader:
        """Same as :meth:`_get_val_loader` (kept separate for clarity in overrides)."""
        return DataLoader(
            dataset=data,
            shuffle=self.shuffle,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers if not self.debug else 0,
            collate_fn=collate_fn,
            pin_memory=(not self.debug),
            persistent_workers=(not self.debug)
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """One loader per benchmark in ``test_data_list``."""
        return [self._get_val_loader(test_data) for test_data in self.test_data_list]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """One loader per benchmark in ``test_data_list``."""
        return [self._get_test_loader(test_data) for test_data in self.test_data_list]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Defaults to RoadAnomaly only."""
        return DataLoader(
            dataset=self.roadanomaly,
            shuffle=self.shuffle,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers if not self.debug else 0,
            collate_fn=collate_fn,
            pin_memory=(not self.debug),
            persistent_workers=False
        )