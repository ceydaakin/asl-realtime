"""Training-time augmentation on raw landmark x/y, batched on the training device.

Points that were not detected (exact 0) and padded frames stay 0. The dominant
hand is never dropped: it carries most of the sign. Lips and arm points go
missing in real footage (face turned away, arm out of frame), so the model
should learn to cope without them.

Temporal augmentation changes the signing speed and drops frames, the way a
slower phone or a missed hand detection would. Real frames stay packed at the
start of the sequence, so it returns a new mask as well.
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
    time_scale: tuple[float, float] = (0.75, 1.25)  # >1 = slower signing, more frames
    drop_frame_p: float = 0.1


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


def augment_time(xy: torch.Tensor, mask: torch.Tensor, cfg: AugConfig,
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample each sequence to a random speed, then drop random frames. Returns new (xy, mask)."""
    n, t = mask.shape
    pos = torch.arange(t)
    length = mask.sum(dim=1).cpu()
    new_len = (length * _uniform(*cfg.time_scale, n, generator)).round().clamp(min=1, max=t)
    new_len = torch.where(length > 0, new_len, torch.zeros_like(new_len))
    # Nearest source frame, not interpolation: blending a detected point with a missing one (0) is meaningless.
    src = ((pos + 0.5) * (length / new_len.clamp(min=1))[:, None]).long().clamp(max=t - 1)
    keep = (torch.rand(n, t, generator=generator) >= cfg.drop_frame_p) & (pos < new_len[:, None])
    keep[:, 0] = length > 0
    order = (~keep).long().argsort(dim=1, stable=True)  # kept frames first, original order
    idx = src.gather(1, order).to(xy.device)
    new_mask = (pos < keep.sum(dim=1, keepdim=True)).to(mask.device)
    moved = xy.gather(1, idx[:, :, None, None].expand_as(xy))
    return torch.where(new_mask[:, :, None, None], moved, torch.zeros_like(xy)), new_mask
