"""Training-time augmentation on raw landmark x/y, batched on the training device.

Points that were not detected (exact 0) and padded frames stay 0. The dominant
hand is never dropped: it carries most of the sign. Lips and arm points go
missing in real footage (face turned away, arm out of frame), so the model
should learn to cope without them.
"""

import math
from dataclasses import dataclass

import torch

from .landmarks import LIPS_SLICE, N_COLS, POSE_SLICE


@dataclass(frozen=True)
class AugConfig:
    scale: tuple[float, float] = (0.8, 1.2)
    aspect: tuple[float, float] = (0.9, 1.1)   # extra x-only scale
    rotate_deg: float = 15.0
    shear: float = 0.15
    drop_lips_p: float = 0.1
    drop_pose_p: float = 0.1
    drop_point_p: float = 0.05                  # per landmark, for the whole sequence


def _uniform(lo: float, hi: float, n: int, g: torch.Generator | None) -> torch.Tensor:
    return torch.rand(n, generator=g) * (hi - lo) + lo


def _affine(n: int, cfg: AugConfig, g: torch.Generator | None) -> torch.Tensor:
    """(n, 2, 2) = rotation @ shear @ scale."""
    s = _uniform(*cfg.scale, n, g)
    sx = s * _uniform(*cfg.aspect, n, g)
    a = _uniform(-cfg.rotate_deg, cfg.rotate_deg, n, g) * (math.pi / 180)
    sh = _uniform(-cfg.shear, cfg.shear, n, g)
    zero, one = torch.zeros(n), torch.ones(n)
    scale = torch.stack([torch.stack([sx, zero], -1), torch.stack([zero, s], -1)], -2)
    shear = torch.stack([torch.stack([one, sh], -1), torch.stack([zero, one], -1)], -2)
    rot = torch.stack([torch.stack([a.cos(), -a.sin()], -1), torch.stack([a.sin(), a.cos()], -1)], -2)
    return rot @ shear @ scale


def _keep_mask(n: int, cfg: AugConfig, g: torch.Generator | None) -> torch.Tensor:
    """(n, N_COLS) True for landmarks that survive dropout."""
    keep = torch.rand(n, N_COLS, generator=g) >= cfg.drop_point_p
    keep[:, LIPS_SLICE] &= (torch.rand(n, 1, generator=g) >= cfg.drop_lips_p)
    keep[:, POSE_SLICE] &= (torch.rand(n, 1, generator=g) >= cfg.drop_pose_p)
    return keep


def augment(xy: torch.Tensor, mask: torch.Tensor, cfg: AugConfig,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """xy: (B, T, N_COLS, 2) in image coordinates. Returns a new tensor, same shape."""
    n = xy.shape[0]
    present = (xy != 0).any(dim=-1, keepdim=True) & mask[:, :, None, None]
    m = _affine(n, cfg, generator).to(xy.device, xy.dtype)
    keep = _keep_mask(n, cfg, generator).to(xy.device)[:, None, :, None]
    moved = torch.einsum("btcd,bed->btce", xy - 0.5, m) + 0.5
    return torch.where(present & keep, moved, torch.zeros_like(xy))
