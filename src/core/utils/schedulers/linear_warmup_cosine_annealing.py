"""OneCycleLR variant with per-parameter-group ``lr_scale`` support."""

import warnings

from torch.optim.lr_scheduler import OneCycleLR


class LinearWarmupCosineAnnealingLR(OneCycleLR):
    """Linear warmup, optional hold, then cosine decay (PyTorch ``OneCycleLR`` schedule).

    Per-group learning rates are scaled by ``param_group['lr_scale']`` (default ``1.0``),
    which ``FlowClasModule`` sets when building optimizer param groups.
    """

    def __init__(self, optimizer, max_lr, lr_scale=None, **kwargs):
        # ``lr_scale`` is accepted for CLI/config compatibility; scaling uses param groups.
        super().__init__(optimizer, max_lr, **kwargs)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num} times. "
                f"The specified number of total steps is {self.total_steps}"
            )

        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase["end_step"]
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    if i < len(self._schedule_phases) - 1:
                        self.anneal_func = self._annealing_linear
                    else:
                        self.anneal_func = self._annealing_cos
                    scale_factor = group.get("lr_scale", 1.0)
                    computed_lr = self.anneal_func(
                        group[phase["start_lr"]] * scale_factor,
                        group[phase["end_lr"]] * scale_factor,
                        pct,
                    )
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func(
                            group[phase["start_momentum"]],
                            group[phase["end_momentum"]],
                            pct,
                        )
                    break
                start_step = phase["end_step"]

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    group["betas"] = (computed_momentum, *group["betas"][1:])
                else:
                    group["momentum"] = computed_momentum

        return lrs
