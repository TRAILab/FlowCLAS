"""Likelihood-based anomaly map: marginalize mixture components, invert to score, optional blur."""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, nn


class AnomalyLikelihoodMap(nn.Module):
    """Map per-scale log-densities to a single (B, 1, H, W) anomaly score in [0, 1]."""

    def __init__(self, use_gaussian: bool = False) -> None:
        super().__init__()
        self.use_gaussian = use_gaussian

    def forward(self, out: dict[str, Tensor], shape: tuple[int, int]) -> Tensor:
        """Aggregate ``out["log_p"]`` across scales; higher score means more anomalous.

        Args:
            out: Model output containing ``log_p`` (list of per-scale log-prob tensors).
            shape: Target spatial size ``(H, W)`` matching the input image.

        Returns:
            Tensor of shape ``(N, 1, H, W)``.
        """
        if "log_p" not in out:
            raise KeyError("Expected key 'log_p' in model output.")

        likelihoods: list[Tensor] = []
        for lp in out["log_p"]:
            prob = torch.exp(lp)
            prob = torch.nan_to_num(prob, nan=0.0)
            # Marginalize mixture / channel components: sum_c p(x|c) when log_p has C>1.
            prob = torch.sum(prob, dim=1, keepdim=True)
            ll = F.interpolate(prob, size=shape, mode="bilinear", align_corners=False)
            if self.use_gaussian:
                ll = TF.gaussian_blur(ll, kernel_size=7)
            likelihoods.append(ll)

        scores = 1.0 - torch.mean(torch.stack(likelihoods, dim=-1), dim=-1)
        if torch.isnan(scores).any():
            raise ValueError("Anomaly map contains NaN values.")
        return scores
