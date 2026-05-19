"""Segmentation metrics for training-time logging."""

import torch
from torch import Tensor


def accuracy(pred: Tensor, target: Tensor) -> Tensor:
    """Pixel-wise semantic segmentation accuracy as a percentage.

    Args:
        pred: Class logits, shape (N, C, H, W).
        target: Ground-truth class indices, shape (N, H, W).

    Returns:
        Scalar tensor with the fraction of correct pixels, scaled to [0, 100].
    """
    _, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))

    eps = torch.finfo(torch.float32).eps
    total_correct = correct[0].reshape(-1).float().sum(0, keepdim=True) + eps
    total_num = target.numel() + eps
    return total_correct.mul_(100.0 / total_num)
