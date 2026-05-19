"""ActNorm for normflows-style stacks (used inside ``ConvNet2d`` when ``actnorm=True``)."""

import math

import torch
from torch import Tensor, nn


class ActNorm(nn.Module):
    """Per-channel scale and shift; first batch performs data-dependent init (cf. batch norm)."""

    def __init__(self, shape: tuple[int, ...], scale: bool = True, shift: bool = True) -> None:
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("s", torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()
        self.register_buffer("data_dep_init_done", torch.tensor(0.0))

    def forward(self, z: Tensor) -> Tensor:
        if not self.data_dep_init_done:
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.0, device=z.device)
        return z * torch.exp(self.s) + self.t

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        if not self.data_dep_init_done:
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0, device=z.device)

        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = math.prod(z.size(i) for i in self.batch_dims[1:])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det
