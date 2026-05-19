"""Base distributions for flow models."""

import torch
from torch import Tensor, nn


class MultivariateNormal(nn.Module):
    """Diagonal Gaussian in ``dim`` dims; ``mu`` / ``log_scale`` are global (learned)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("log_normalizer", -0.5 * dim * torch.log(torch.tensor(2 * torch.pi)))

    def log_prob(
        self,
        x: Tensor,
        return_z: bool = False,
        true_logp: bool = False,
        sum_dim: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        """Return log p(x) with same spatial layout as ``x`` (``B,1,H,W`` when ``x`` is 4D)."""
        del sum_dim, kwargs
        mu = self.mu[None, :]
        log_scale = self.log_scale[None, :]

        if x.dim() == 4:
            b, _, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b * h * w, self.dim)
        else:
            b = h = w = 0
            x_flat = x

        z = (x_flat - mu) * torch.exp(-log_scale)

        if self.training or true_logp:
            log_p = self.log_normalizer - self.log_scale.sum() - 0.5 * torch.pow(z, 2).sum(dim=1, keepdim=True)
        else:
            log_p = torch.mean(-0.5 * torch.pow(z, 2) - self.log_scale, dim=1, keepdim=True)

        if x.dim() == 4:
            z_spatial = z.reshape(b, h, w, self.dim).permute(0, 3, 1, 2)
            log_p = log_p.reshape(b, h, w, 1).permute(0, 3, 1, 2)
        else:
            z_spatial = z

        if return_z:
            return log_p, z_spatial
        return log_p, None

    def sample(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.dim, device=self.mu.device)
        return z * torch.exp(self.log_scale[None, :]) + self.mu[None, :]
