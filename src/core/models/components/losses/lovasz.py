"""Lovasz loss for segmentation tasks."""

import torch


"""Cross Entropy Loss."""


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszLoss(nn.Module):
    """Lovasz loss for segmentation."""
    def __init__(self, num_classes: int, ignore_index: int = -100) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, seg_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the Lovasz loss for segmentation.
        Args:
            seg_logits (torch.Tensor): Logits from the decode head. Shape (N, C, H, W).
            target (torch.Tensor): Target tensor. Shape (N, H, W).
        Returns:
            torch.Tensor: Lovasz loss.
        """
        probs = F.softmax(seg_logits, dim=1)    # (N, C, H, W)
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)    # (N*H*W, C)
        target = target.view(-1)    # (N*H*W,)

        valid = target != self.ignore_index
        probs = probs[valid]
        target = target[valid]

        if valid.sum() == 0:
            return torch.tensor(0.0, device=seg_logits.device)

        loss = torch.tensor(0.0, device=seg_logits.device)
        assert probs.shape[-1] == self.num_classes, f"in lovasz loss: {probs.shape} != {self.num_classes}"
        for c in range(self.num_classes):
            fg = (target == c).float()
            if fg.sum() == 0:
                continue
            if self.num_classes == 1:
                raise NotImplementedError("Not implemented for binary segmentation.")
            class_pred = probs[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            loss += torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted)))
        return loss / self.num_classes