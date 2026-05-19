"""FrEIA ActNorm subclass used by FlowCLAS (device-safe init)."""

import torch
from FrEIA.modules import ActNorm as _ActNorm


class ActNorm(_ActNorm):
    """FrEIA ActNorm with safer ``initialize`` when some channels have zero std."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize(self, batch: torch.Tensor) -> None:
        if batch.ndim != self.log_scale.ndim:
            raise ValueError(f"Expected batch of dimension {self.log_scale.ndim}, but got {batch.ndim}.")

        dims = [0] + list(range(2, batch.ndim))
        loc = torch.mean(batch, dim=dims, keepdim=True)
        scale = torch.std(batch, dim=dims, keepdim=True)

        invalid = torch.isclose(scale, torch.tensor(0.0, dtype=scale.dtype, device=scale.device))
        if torch.any(invalid):
            scale = scale.clone()
            scale[invalid] = torch.mean(scale[~invalid])

        self.log_scale.data[:] = torch.log(scale)
        self.loc.data[:] = loc
        self.is_initialized.data = torch.tensor(True, device=self.is_initialized.device)

    def forward(self, x, c=None, rev=False, jac=True, **kwargs):
        return super().forward(x, rev=rev, jac=jac)
