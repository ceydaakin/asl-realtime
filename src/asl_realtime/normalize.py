"""Landmark normalization as a torch module, so it ships inside the exported model.

The on-device model then takes raw MediaPipe x/y (plus the frame mask) and the
app never has to re-implement preprocessing.

Modes:
  sequence  center on the mean detected point of the whole sequence and divide by
            its spread. Removes where the signer stands and how far they are
            from the camera.
  global    per-landmark mean/std from the training set (absolute image position).
"""

import torch
from torch import nn

from .features import FeatureStats, USE_DIMS
from .landmarks import N_COLS

MODES = ("sequence", "global")


class Normalizer(nn.Module):
    def __init__(self, mode: str = "sequence", stats: FeatureStats | None = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")
        if mode == "global" and stats is None:
            raise ValueError("global mode needs training stats")
        self.mode = mode
        if mode == "global":
            self.register_buffer("mean", torch.as_tensor(stats.mean, dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(stats.std, dtype=torch.float32))

    def forward(self, xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """xy: (B, T, N_COLS, 2), exact 0 = not detected. mask: (B, T). -> (B, T, N_COLS * 2)."""
        present = ((xy != 0).any(dim=-1, keepdim=True) & mask[:, :, None, None]).to(xy.dtype)
        if self.mode == "global":
            z = (xy - self.mean) / self.std
        else:
            count = present.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            center = (xy * present).sum(dim=(1, 2), keepdim=True) / count
            spread = (((xy - center) * present) ** 2).sum(dim=(1, 2, 3), keepdim=True) / (count * USE_DIMS)
            z = (xy - center) / spread.sqrt().clamp(min=1e-6)
        return (z * present).reshape(xy.shape[0], xy.shape[1], N_COLS * USE_DIMS)
