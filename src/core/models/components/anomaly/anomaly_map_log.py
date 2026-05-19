"""Log-density anomaly map: mean negative log-probability across scales, optional blur."""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, nn


class AnomalyLogLikelihoodMap(nn.Module):
    """Map per-scale log p(z) to a single (B, 1, H, W) score (higher = more anomalous)."""

    def __init__(self, use_gaussian: bool = False) -> None:
        super().__init__()
        self.use_gaussian = use_gaussian

    def forward(self, out: dict[str, Tensor], shape: tuple[int, int]) -> Tensor:
        """Interpolate each scale's log-density to ``shape``, optionally blur, then mean and negate.

        Args:
            out: Model output containing ``log_p`` (list of per-scale log-prob tensors).
            shape: Target spatial size ``(H, W)`` matching the input image.

        Returns:
            Tensor of shape ``(N, 1, H, W)``.
        """
        if "log_p" not in out:
            raise KeyError("Expected key 'log_p' in model output.")

        maps: list[Tensor] = []
        for lp in out["log_p"]:
            ll = F.interpolate(lp, size=shape, mode="bilinear", align_corners=False)
            if self.use_gaussian:
                ll = TF.gaussian_blur(ll, kernel_size=7)
            maps.append(ll)
        scores = -torch.mean(torch.stack(maps, dim=-1), dim=-1)
        if torch.isnan(scores).any():
            raise ValueError("Anomaly map contains NaN values.")
        return scores
