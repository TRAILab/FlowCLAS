"""Implementation of FusedTransforms class for combining albumentations and torchvision transforms."""


from __future__ import annotations

import random
from abc import ABC
from enum import Enum

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.v2 as v2

from anomalib.data.utils.image import get_image_height_and_width

from core.utils.pre_processing import generate_polygon


class InputNormalizationMethod(str, Enum):
    """Which channel statistics :class:`FusedTransforms` uses for ``torchvision`` normalize."""

    IMAGENET = "imagenet"
    ALLO = "allo"
    CLIP = "clip"


class NormalizationStats(Enum):
    """Per-method mean/std tuples consumed by :class:`FusedTransforms`."""
    IMAGENET = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }
    ALLO = {
        "mean": (0.2083, 0.2124, 0.2146),
        "std": (0.2669, 0.2700, 0.2711)
    }
    CLIP = {
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711)
    }


class RandomScaleV2(A.RandomScale):
    """``A.RandomScale`` that optionally samples only from ``scale_factors``."""

    def __init__(self, scale_factors: list[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.scale_factors = scale_factors

    def get_params(self) -> dict[str, float]:
        if self.scale_factors is not None:
            scale = random.choice(self.scale_factors)
            if not isinstance(scale, float):
                raise ValueError(f"Invalid scale factor: {scale}")
            return {"scale": scale}

        return super().get_params()


class PhotoMetricDistortion(A.Compose):
    """MMSeg-style color jitter stack wrapped as a single stochastic compose."""

    def __init__(self, p: float = 0.5, **kwargs):
        transforms = [
            A.ColorJitter(brightness=0.12, contrast=0.0, saturation=0.0, hue=0.0, p=0.5),
            A.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.0, hue=0.0, p=0.5),
            A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0, p=0.5),
        ]
        super().__init__(transforms, p=p, **kwargs)

class FusedTransforms(ABC):
    """Albumentations augmentations followed by resize/pad, tensor cast, and optional normalize.

    Pipeline:
        #. Optional synthetic anomaly injection (``synthesize`` flag).
        #. User ``albumentations`` compose.
        #. ``LongestMaxSize`` / divisor padding when configured.
        #. ``ToTensorV2`` then ``torchvision`` dtype + normalize.
    """

    def __init__(
        self,
        transforms: A.Compose | None,
        image_size: int | tuple[int, int] | None,
        normalization: InputNormalizationMethod,
        normalize: bool = True,
        hist_eq: bool = False,
        pad_value: float | tuple[float] = 0.0,
        mask_pad_value: int = -100,
        divisor: int | None = None,
        synthesize_cfg: dict | None = None,
    ) -> None:
        if synthesize_cfg is None:
            synthesize_cfg = {}
        self.albu_transforms = transforms if transforms is not None else A.Compose([])
        self.image_size = get_image_height_and_width(image_size) if image_size is not None else None
        self.normalization = normalization
        self.normalize = normalize
        self.hist_eq = hist_eq
        if isinstance(pad_value, float):
            pad_value = (pad_value, pad_value, pad_value)
        self.pad_value = pad_value
        self.mask_pad_value = mask_pad_value
        self.divisor = divisor
        self.synthesize_cfg = synthesize_cfg
        self._synthesize = False

    @property
    def synthesize(self) -> bool:
        return self._synthesize

    @synthesize.setter
    def synthesize(self, value: bool) -> None:
        self._synthesize = value

    @property
    def resize(self) -> bool:
        geo_transforms = [A.RandomResizedCrop, A.RandomScale, A.LongestMaxSize, A.RandomCrop]
        return not any(type(t) in geo_transforms for t in self.albu_transforms)

    def __call__(self, image, *args, **kwargs):
        """Apply fused transforms; pass a mask as the first extra arg to transform image+mask together."""
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        else:
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Unknown input image type: {type(image)}")

        if getattr(self, "hist_eq", False):
            raise NotImplementedError("Histogram equalization is not implemented.")

        input_dict = {"image": image}
        if args:
            mask = args[0]
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 3:
                    mask = mask[0]
                mask = mask.numpy()
            else:
                if not isinstance(mask, np.ndarray):
                    raise ValueError(f"Unknown input mask type: {type(mask)}")
            input_dict["mask"] = mask

        transformed = (
            self.generate_synthetic_anomaly(input_dict) if self.synthesize else input_dict
        )

        transformed = self.albu_transforms(**transformed) if self.albu_transforms is not None else input_dict

        transformed = (
            self.resize_transform(**transformed) if self.resize_transform is not None else transformed
        )

        transformed = ToTensorV2()(**transformed)

        final_img = self.post_transforms(transformed["image"])
        
        if "mask" in transformed:
            return final_img, transformed["mask"]
        else:
            return final_img

    @property
    def norm_stats(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(mean, std)`` tuples for the configured :class:`InputNormalizationMethod`."""
        if self.normalization == InputNormalizationMethod.IMAGENET:
            stats = NormalizationStats.IMAGENET
        elif self.normalization == InputNormalizationMethod.ALLO:
            stats = NormalizationStats.ALLO
        elif self.normalization == InputNormalizationMethod.CLIP:
            stats = NormalizationStats.CLIP
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
        
        mean = stats.value["mean"]
        std = stats.value["std"]
        return mean, std

    @property
    def post_transforms(self) -> v2.Transform:
        """``ToDtype`` to float32 in ``[0, 1]`` plus optional ``Normalize``."""
        mean, std = self.norm_stats
        transforms = [
            v2.ToDtype(torch.float32, scale=True)
        ]
        if self.normalize:
            transforms.append(
                v2.Normalize(mean=mean, std=std)
            )
        return v2.Compose(transforms)

    @property
    def normalize_transform(self) -> v2.Normalize:
        """Image-only ``Normalize`` using the same statistics as :meth:`norm_stats`."""
        mean, std = self.norm_stats
        return v2.Normalize(mean=mean, std=std)

    @property
    def resize_transform(self) -> A.Compose:
        """Longest-side resize plus optional divisor padding (no padding to fixed H×W unless configured)."""
        transformations = []
        if self.image_size is not None and self.resize:
            transformations.extend(
                [
                    A.LongestMaxSize(
                        max_size=self.image_size[1],
                        interpolation=3,
                        always_apply=True,
                    ),
                ]
            )
        if self.divisor is not None and isinstance(self.divisor, int):
            transformations.append(
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=self.divisor,
                    pad_width_divisor=self.divisor,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=self.pad_value,
                    mask_value=self.mask_pad_value,
                    always_apply=True,
                )
            )
        return A.Compose(transformations)

    def generate_synthetic_anomaly(self, input_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Paint random multivariate-normal patches (in normalized color space) on ``image``/``mask``."""
        image = input_dict["image"]
        mask = input_dict["mask"]
        H, W = mask.shape
        num_objs = np.random.randint(0, int(self.synthesize_cfg["max_num_objs"]) + 1, (1,)).item()
        num_points = int(self.synthesize_cfg["num_points"])
        min_factor, max_factor = float(self.synthesize_cfg["min_factor"]), \
                                    float(self.synthesize_cfg["max_factor"])

        for _ in range(num_objs):
            rr, cc, box_w, box_h = generate_polygon(H, W, num_points, (min_factor, max_factor))
            num_outliers = len(rr)
            mean, std = self.norm_stats
            mean_arr, cov = np.array(mean), np.diag(np.array(std) ** 2)
            noise = np.random.multivariate_normal(mean_arr, cov, (num_outliers,))

            x = np.random.randint(0, W - box_w - 1, (1,))
            y = np.random.randint(0, H - box_h - 1, (1,))
            rr += x
            cc += y

            image[cc, rr, :] = noise
            mask[cc, rr] = self.synthesize_cfg["anomaly_label"]

        return {"image": image, "mask": mask}