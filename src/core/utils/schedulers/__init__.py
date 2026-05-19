"""Custom LR schedulers registered for LightningCLI ``class_path`` resolution."""

from .linear_warmup_cosine_annealing import LinearWarmupCosineAnnealingLR

__all__ = ["LinearWarmupCosineAnnealingLR"]
