"""Pixel-wise InfoNCE contrastive losses for two training regimes.

- ``MixOODContrastLoss``: separate projections for the mix batch and the OOD batch
  (``forward`` takes four tensors). Negatives can include pixels from the real OOD image.
- ``MixOnlyContrastLoss``: a single projection map and mask (``forward`` takes two tensors);
  used when there is no OOD image stream (``no_ood`` training).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def _info_nce(
    a_samples: torch.Tensor,
    a_labels: torch.Tensor,
    c_samples: torch.Tensor,
    c_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE / NT-Xent loss given anchor and contrast sets (same math as SimCLR)."""
    mask = torch.eq(a_labels, c_labels.transpose(0, 1)).float()
    anchor_dot_contrast = (a_samples @ c_samples.transpose(0, 1)) / temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    neg_mask = 1 - mask
    mask = mask.fill_diagonal_(0.0)
    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits + neg_logits)
    pos_count = mask.sum(1).clamp(min=1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count
    return -mean_log_prob_pos.mean()


class MixOODContrastLoss(nn.Module):
    """InfoNCE using both in-distribution (mix) and OOD image projections."""

    def __init__(self, temperature: float, ood_idx: int, ignore_idx: int = -100, max_views: int = 512):
        super().__init__()
        self.temperature = temperature
        self.ignore_idx = ignore_idx
        self.ood_idx = ood_idx
        self.max_views = max_views

    def forward(self, 
                mix_proj: torch.Tensor,
                mix_gt: torch.Tensor,
                ood_proj: torch.Tensor,
                ood_gt: torch.Tensor):

        mix_gt = F.interpolate(mix_gt.unsqueeze(1).float(), 
                                size=mix_proj.shape[2:],
                                mode='nearest').squeeze(1).long()
        ood_gt = F.interpolate(ood_gt.unsqueeze(1).float(), 
                                size=ood_proj.shape[2:],
                                mode='nearest').squeeze(1).long()

        #? Normalize the embeddings for contrastive learning
        mix_proj = F.normalize(mix_proj, p=2, dim=1)
        ood_proj = F.normalize(ood_proj, p=2, dim=1)

        #? Randomly sample the anchor and contrastive samples
        (anchor_embeds, anchor_labels,
         contrast_embeds, contrast_labels) = self.get_anchor_con_samples(mix_proj, mix_gt, ood_proj, ood_gt)

        #? calculate the InfoNCE
        if anchor_embeds.numel() > 0:
            loss = _info_nce(
                anchor_embeds,
                anchor_labels.unsqueeze(1),
                contrast_embeds,
                contrast_labels.unsqueeze(1),
                self.temperature,
            )
        else:
            loss = torch.tensor(0., device=mix_proj.device)
        return loss

    def get_anchor_con_samples(self, oe_embeds, oe_labels, ood_embeds, ood_labels):
        #? Reformat for ease of sampling
        oe_embeds = oe_embeds.flatten(start_dim=2).permute(0, 2, 1)
        oe_labels = oe_labels.flatten(start_dim=1)
        ood_embeds = ood_embeds.flatten(start_dim=2).permute(0, 2, 1)
        ood_labels = ood_labels.flatten(start_dim=1)

        #? define different types of embeds
        oe_pos = oe_embeds[oe_labels == self.ood_idx]
        oe_neg = oe_embeds[(oe_labels != self.ood_idx) & (oe_labels != self.ignore_idx)]
        ood_pos = ood_embeds[ood_labels == self.ood_idx]
        ood_neg = ood_embeds[(ood_labels != self.ood_idx) & (ood_labels != self.ignore_idx)]

        #? check if ood_neg is empty (probably if background of ood image is ignored explicitly)
        has_ood_neg = ood_neg.numel() > 0
        num_sample_list = [self.max_views, oe_pos.shape[0], oe_neg.shape[0], ood_pos.shape[0]]

        if has_ood_neg:
            num_sample_list.append(ood_neg.shape[0])

        #? define the number of samples
        sample_num = int(min(*num_sample_list))

        #? randomly extract the anchor set with {oe_pos, oe_neg}
        anchor_pos = oe_pos[torch.randperm(oe_pos.shape[0])][:sample_num]
        anchor_neg = oe_neg[torch.randperm(oe_neg.shape[0])][:sample_num]
        anchor_set = torch.cat([anchor_pos, anchor_neg], dim=0)
        anchor_label = torch.cat([torch.ones(anchor_pos.shape[0], device=anchor_pos.device),
                                  torch.zeros(anchor_neg.shape[0], device=anchor_neg.device)])

        #? randomly extract the contrastive set with {oe_pos, oe_neg, ood_pos, ood_neg}
        ood_pos_con = ood_pos[torch.randperm(ood_pos.shape[0])][:sample_num]
        ood_neg_con = ood_neg[torch.randperm(ood_neg.shape[0])][:sample_num]
        contrast_set = torch.cat([anchor_set.clone(), ood_pos_con, ood_neg_con], dim=0)
        contrast_label = torch.cat([anchor_label.clone(),
                                    torch.ones(ood_pos_con.shape[0], device=ood_pos_con.device),
                                    torch.zeros(ood_neg_con.shape[0], device=ood_neg_con.device)])

        return anchor_set, anchor_label, contrast_set, contrast_label


class MixOnlyContrastLoss(nn.Module):
    """InfoNCE on one feature map: positives/negatives mined only from the mix image."""

    def __init__(
        self,
        temperature: float,
        ood_idx: int,
        ignore_idx: int = -100,
        max_views: int = 512,
    ):
        super().__init__()
        self.temperature = temperature
        self.ignore_idx = ignore_idx
        self.ood_idx = ood_idx
        self.max_views = max_views

    def forward(self, embeds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt_ = (
            F.interpolate(gt.unsqueeze(1).float(), size=embeds.shape[2:], mode="nearest")
            .squeeze(1)
            .long()
        )
        anchor_embeds, anchor_labels = self.get_anchor_con_samples(embeds, gt_)
        if anchor_embeds.numel() > 0:
            return _info_nce(
                anchor_embeds,
                anchor_labels.unsqueeze(1),
                anchor_embeds,
                anchor_labels.unsqueeze(1),
                self.temperature,
            )
        return torch.tensor(0.0, device=embeds.device)

    def get_anchor_con_samples(self, embeds: torch.Tensor, labels: torch.Tensor):
        embeds = embeds.flatten(start_dim=2).permute(0, 2, 1)
        labels = labels.flatten(start_dim=1)
        pos = embeds[labels == self.ood_idx]
        neg = embeds[(labels != self.ood_idx) & (labels != self.ignore_idx)]
        sample_num = int(min(self.max_views, pos.shape[0], neg.shape[0]))
        anchor_pos = pos[torch.randperm(pos.shape[0], device=pos.device)][:sample_num]
        anchor_neg = neg[torch.randperm(neg.shape[0], device=neg.device)][:sample_num]
        anchor_set = torch.cat([anchor_pos, anchor_neg], dim=0)
        anchor_label = torch.cat(
            [
                torch.ones(anchor_pos.shape[0], device=anchor_pos.device),
                torch.zeros(anchor_neg.shape[0], device=anchor_neg.device),
            ]
        )
        return anchor_set, anchor_label
