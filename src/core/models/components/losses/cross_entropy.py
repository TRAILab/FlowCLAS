"""Cross Entropy Loss."""


import torch
from torch import nn
import torch.nn.functional as F


class CESegLoss(nn.Module):
    """Cross entropy loss for segmentation.
    Args:
        binary (bool): Whether to use binary classification or multi-class classification.
        use_weights(bool, optional): Whether to use class weights. Defaults to True.
            See https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    """
    def __init__(self, use_weights: bool = True, gamma: float = 0.0, ignore_index: int = -100) -> None:
        super().__init__()
        self.use_weights = use_weights
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, 
                seg_logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Calculate the cross entropy loss for segmentation.
        Args:
            seg_logits (torch.Tensor): Logits from the decode head. Shape (N, C, H, W).
            target (torch.Tensor): Target tensor. Shape (N, H, W).
        Returns:
            torch.Tensor: Cross entropy loss.
        """
        if seg_logits.shape[1] == 1:
            #* Binary classification
            logits = seg_logits.squeeze(1)
            if self.use_weights:
                weights = torch.ones_like(target, dtype=logits.dtype)
                ratio = (target == 0).sum(dim=(1,2), keepdim=True) / (target == 1).sum(dim=(1,2), keepdim=True)
                ratio = ratio.expand_as(weights).to(weights.dtype)
                weights[target == 1] = ratio[target == 1]
            else:
                weights = None
            decode_loss = F.binary_cross_entropy_with_logits(
                seg_logits, target.float(), pos_weight=weights, reduction='mean')
        else:
            #* Multi-class classification
            if self.gamma != 0.0:
                log_p = F.log_softmax(seg_logits, dim=1)
                ce = F.nll_loss(log_p, target.long(), reduction='none', ignore_index=self.ignore_index)
                p = torch.exp(-ce)  #* Directly compute p from ce
                decode_loss = ((1 - p).pow(self.gamma) * ce).mean()
            else:
                decode_loss = F.cross_entropy(seg_logits, target.long(), reduction='mean', ignore_index=self.ignore_index)

        return decode_loss