"""Mask loading helpers."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2.functional import to_image
from torchvision.tv_tensors import Mask

import mmcv
import mmengine.fileio as fileio


def read_mask(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Load a single-channel mask from ``path`` as ``uint8`` numpy or ``Mask`` tensor."""
    image = Image.open(path).convert("L")
    if not as_tensor:
        return np.array(image)
    image = to_image(image).squeeze()
    return Mask(image, dtype=torch.uint8)


def read_mask_mmseg(path: str | Path) -> np.ndarray:
    """Load mask bytes through mmengine and decode with mmcv (uint8 ``H×W``)."""
    img_bytes = fileio.get(path, backend_args=None)
    return mmcv.imfrombytes(img_bytes, flag="unchanged", backend="pillow").squeeze().astype(np.uint8)