"""FlowCLAS pixel NLL: change of variables on inlier pixels (mask == 0).

Shipped flowclas configs use a single Gaussian base component per scale only.
Contrastive loss lives in ``core.models.components.losses.MixOODContrastLoss``.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FlowCLASNLLLoss(nn.Module):
    """NLL from flow Jacobians and base log-densities on inlier pixels (binary mask channel 0)."""

    def __init__(self, reduce: bool = False, ignore_index: int = 255) -> None:
        super().__init__()
        self.reduce = reduce
        self.ignore_index = ignore_index

    def forward(
        self,
        jacobians: list[torch.Tensor],
        log_ps: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=jacobians[0].device)  # pylint: disable=not-callable

        for jacobian, log_p in zip(jacobians, log_ps, strict=True):
            mask_ = (
                F.interpolate(mask.unsqueeze(1).float(), size=log_p.shape[2:], mode="nearest-exact")
                .squeeze(1)
                .to(dtype=torch.long)
            )
            n_pix = torch.numel(log_p[0].detach())
            valid = (mask_.unsqueeze(1) == 0) & (mask_.unsqueeze(1) != self.ignore_index)
            jacobian = jacobian / n_pix * valid.sum(dim=(1, 2, 3))
            loss_ = torch.mean((valid * log_p).sum(dim=(1, 2, 3)) + jacobian)
            loss -= loss_ / n_pix if self.reduce else loss_

        return loss
